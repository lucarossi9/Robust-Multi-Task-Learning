from DatasetGenerator import *
from Optimizer import *
from Utils import *
import seaborn as sns
sns.set_theme()
import pickle

np.random.seed(42)  # fix seed

# the parameters used for most of our experiments
n_tasks = 200
task_size = 25
high_dim = 50  # ---->  dimension of the space (d in the report)
low_dim = 5  # ---->  rank (r in the report)
sparsity_type = "L12"
sparsity = 4
# sparsity_type="entrywise"
# sparsity = (high_dim * n_tasks) * 0.08
save_res = False

# dataset creation
generator = DataGenerator(low_rank_dist=np.random.normal, sparse_dist=np.random.normal, noise_dist=np.random.normal,
                          features_dist=np.random.normal, label_noise_dist=np.random.normal,
                          sparsity_type=sparsity_type,
                          noise_scale=1., noise_flag=True)

dataset = generator.return_dataset(n_tasks, high_dim, low_dim, task_size, sparsity)
print("GENERATED DATASET")

# set optimizer
optimizer = Optimizers(dataset["features"], dataset["labels"])

# set initial iterates and parameters
low_rank_mat = np.zeros((high_dim, n_tasks))
sparse_mat = np.zeros((high_dim, n_tasks))

lambda_sparse = 0.2
lambda_low_rank = 2
n_iters = 10000
t0 = time.time()
results = optimizer.FW_T(sparse_mat, low_rank_mat, lambda_sparse, lambda_low_rank, dataset["features"],
                         dataset["labels"], n_iters, 0.01)  # 0.016
t1 = time.time()
print("time taken=", t1 - t0)
print("FINISHED OPTIMIZING")

# ANALYSIS RESULTS
ratio_of_recovery_analysis(dataset, results)
matrices_info(results, dataset, sparsity, low_dim)
