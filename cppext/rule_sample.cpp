#ifndef NOBIND
#include <torch/extension.h>
#include <ATen/ParallelOpenMP.h>
namespace py = pybind11;
#endif

#include <cstdio>
#include <map>
#include <vector>
#include <set>
#include <thread>
#include <unordered_map>
#include <random>
// #include <mutex>
#include <shared_mutex>
#include <ctime>
#include <cstdlib>
using namespace std;

#ifndef NOBIND
	#define NOBIND
	#include "groundings.cpp"
	#undef NOBIND
#else
	#include "groundings.cpp"
#endif

using cppext_groundings::GroundingsCountTask;

#define IL inline
#define debug // printf
#define ckpt() //fprintf(stderr, "Thread %d Checkpoint: %d\n", id, __LINE__)
#define ckpt_lock() //fprintf(stderr, "Thread %d Lock: %d\n", id, __LINE__)
#define ckpt_unlock() //fprintf(stderr, "Thread %d Unlock: %d\n", id, __LINE__)

namespace cppext_rule_sample {
	using ull = unsigned long long;

	Graph G, Ginv;
	int E, R;
	void add_data(int h, int r, int t) {
		G.add(h, r, t);
		Ginv.add(t, r, h);
	}


	int MAX_LEN, print_round, num_round;
	map<int, int> len;//
	vector<pair<int, vector<int>>> tri;


	map<vector<int>, pair<double, double>> rules;
	shared_mutex rules_mutex;

	struct Choice {//choices[规则长度][h] = [{r',i,抽样次数}...]表示从r关联的某tail i到h的一条可能路径
		int r, q;
		long long c;
	};

	unsigned long long seed(int id = 1) {
		char *a = new char, *b = new char;
		auto ret = b - a;
		delete a;
		delete b;
		ret = ret + 1004535809ll * time(0) + clock();
		while(id--) ret = ret * (1 << 16 | 3) + 33333331;
		return ret;
	}

	void work(const int id, GroundingsCountTask gct) {
		auto sd = seed(id);
		// printf("Thread %d: Start seed = %llX.\n", id, sd);
		ckpt();
		default_random_engine e(sd);
		uniform_int_distribution<long long> rand(0, 1ll << 62);

		ckpt();
		// printf("num_round = %d MAX_LEN = %d\n", num_round, MAX_LEN);
		for(int round = 1; round <= num_round; ++round) {
			auto &cur = tri[rand(e) % tri.size()];//随机选取关系r的一个三元组的(h,t)
			auto h = cur.first;//auto可以在声明变量的时候根据变量初始值的类型自动为此变量选择匹配的类型,必须初始化
			auto t = cur.second;

			vector<unordered_map<int, long long>> path_cnt(MAX_LEN + 1);
			vector<unordered_map<int, vector<Choice>>> choices(MAX_LEN + 1);

			ckpt();
			for(auto i : t) {//遍历r的所有tail，初始化choices
				path_cnt[0][i] = 1;//到达该tail的path的计数
				choices[0][i].push_back({-1, -1, 1});//到达该tail的path，{-1, -1, 1}是终结符
			} 

			ckpt();
			//抽样次数，指节点i作为head的所有可能的路径的出现次数(通过将各个上游节点的抽样次数加和而得到)
			for(int _ = 0; _ < MAX_LEN; ++_) {//遍历所有可能的规则长度，将所有长度小于MAX_LEN的path路径存入choices，抽样次数存入path_cnt
				auto &next_cnt = path_cnt[_ + 1];//声明path_cnt数组的新元素，表示下一个长度的路径上各个i的抽样次数
				auto &next_cho = choices[_ + 1];//声明choices数组的新元素，表示下一个长度的路径上各个i的谓词或实体
				for(auto path : path_cnt[_]) {//遍历path_cnt[_]里的所有元素，作为"_"长度路径tail
					auto i = path.first;//tail的id
					auto cnt = path.second;//tail i的抽样次数
					for(auto edge : Ginv.a[i]) {//G.a[i]是跟head i相关的(r,t),Ginv.a[i]是跟tail i 相关的(r,h)组成list
						next_cnt[edge.second] += cnt;//将tail、目标谓词r所对应的某个h（edge.second）的下一个长度的规则抽样次数加上cnt
						auto &cho = next_cho[edge.second];//tail、目标谓词r所对应的某个h（edge.second）的下一个长度的规则path
						long long last_c = (cho.empty() ? 0 : cho.back().c);
						cho.push_back({edge.first, i, cnt + last_c});//choices[规则长度][h]（tail i到h的path）里添加{r',i,抽样次数}
					}
				}
			}

			ckpt();
			//在choices里寻找head符合(h，目标谓词r,t)的规则。h即为三元组(h,r,t)的head
			//choices记录了从目标谓词r关联的某tail i到h的一条可能路径,形式choices[规则长度][h] = [{r',i,抽样次数}...]
			//len表示每个长度规则的抽样次数？{1: 1, 2: 10, 3: 10, 4: 10}
			for(auto lp : len) {
				int len = lp.first, cnt = lp.second;
				if(path_cnt[len][h] == 0)//无路径可达的h，直接跳过
					continue;
				for(int _c = 0; _c < cnt; ++_c) {//遍历所有抽样次数
					// printf("iter _c = %d\n", _c);
					ckpt();
					vector<int> path;//记载规则
					for(int p = h, l = len; l > 0; --l) {//循环len次，填充该长度的规则
						auto &cho = choices[l][p];//head p跟目标谓词r的某个tail i所连接的长度为l的路径，=[{r',i,抽样次数}...]
						long long k = rand(e) % path_cnt[l][p] + 1;//随机选择一个抽样次数
						int L = 0, R = cho.size() - 1, M;//R表示head p =h到目标谓词r的某个tail i的长度为l的可选路径的数量，一个“choice”
						while(L < R) {
							M = (L + R) >> 1;//左右平均值
							if(k <= cho[M].c) R = M;
							else L = M + 1;//选择一个中间路径点choice作为新的head p =h
						}
						path.push_back(cho[L].r);//将路径上的r'存入path，作为规则的一步
						p = cho[L].q;//以上一个tail作为新的head
					}

					ckpt();
					// ckpt_lock();
					{
						shared_lock<shared_mutex> lock(rules_mutex);
						if(rules.count(path)) continue;//跳过重复规则
					}
					auto iter = rules.end();
					{
						unique_lock<shared_mutex> lock(rules_mutex);
						iter = rules.insert({path, {0.0, 0.0}}).first;//将抽样的路径作为一个规则存入rules
						//对每个round抽取的那个三元组样本，各个长度规则的抽样次数为{1: 1, 2: 10, 3: 10, 4: 10}
					}

					ckpt();
					double recall_u = 0, recall_d = 0, prec_u = 0, prec_d = 0;
					for(auto& t : tri) {//遍历所有三元组，评估规则
						int X = t.first;//三元组head 
						set<int> Y(t.second.begin(), t.second.end());//三元组tail
						recall_d += Y.size();

						// ckpt();
						gct.run(X, path);

						// ckpt();
						for(int i = 0; i < (int) gct.result_pts.size(); ++i) {//遍历path的所有的groudtruth
							auto p = gct.result_pts[i];
							auto c = gct.result_cnt[i];
							prec_d += c;
							if(Y.count(p)) {//规则命中
								prec_u += c;
								recall_u += 1;
							}
						}
					}
					double prec = prec_u / max(0.001, prec_d);
					double recall = recall_u / max(0.001, recall_d);
					iter->second = {prec, recall};//评估规则的先验权重，存入rules

					// printf("found _c = %d path = ", _c);
					// for(auto x : path) printf("%d ", x);
					// printf("recall = %.4lf prec = %.4lf\n", recall, prec);
				}
			}

			ckpt();
			if(round % print_round == 0) 
				printf("Thread %d: Round %d/%d.\n", id, round, num_round);
		}
		// printf("Thread %d: Done.\n", id);

	}

	vector<pair<vector<int>, pair<double, double>>>
	//抽样谓词r相关的所有合理路径作为候选规则，存入数组rules。对每个round抽取的那个三元组样本，各个长度规则的抽样次数为{1: 1, 2: 10, 3: 10, 4: 10}
	run(int r, map<int, int> length_time, int num_samples, 
		int num_threads, double samples_per_print) {
		// printf("Run called\n");
		rules.clear();
		tri.clear();
		for(int i = 0; i < E; ++i)
			if(!G.e[i][r].empty())
				tri.push_back({i, G.e[i][r]});//tri表示谓词r相关的三元组(h,t)

		// printf("tri.size() = %d\n", tri.size());
		//多线程调用work函数，抽样谓词r相关的所有合理路径，存入数组rules。
		if(!tri.empty()) {
			len = length_time;//表示每个规则长度对应的抽样次数？，{1: 1, 2: 10, 3: 10, 4: 10}
			MAX_LEN = 0;
			for(auto a : len) MAX_LEN = max(MAX_LEN, a.first);//规则最大可能的长度
			num_round = num_samples / num_threads;
			print_round = samples_per_print;

			if(num_threads > 1) {
				vector<thread> th(num_threads);
				for(int i = 0; i < num_threads; ++i)
					th[i] = thread(work, i + 1, GroundingsCountTask(G));
				for(int i = 0; i < num_threads; ++i)
					th[i].join();
			} else {
				work(1, GroundingsCountTask(G));
			}
			// printf("All Done!\n");
		}

		return vector<pair<vector<int>, pair<double, double>>>(rules.begin(), rules.end());
	}

}

#ifndef NOBIND
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
	using namespace cppext_rule_sample;
	m.doc() = "Sample Rules for KG";

	m.def("init", [&] (int E, int R) {
		cppext_rule_sample::E = E;
		cppext_rule_sample::R = R;
		G.clear();
		G.init(E, R);
		Ginv.clear();
		Ginv.init(E, R);

	}, py::arg("E"), py::arg("R"));

	m.def("add", add_data, py::arg("h"), py::arg("r"), py::arg("t"));

	m.def("run", run, py::arg("r"), py::arg("length_time"), py::arg("num_samples"),
		py::arg("num_threads"), py::arg("samples_per_print"));
	
}

#endif
#undef IL
#undef debug
#undef ckpt