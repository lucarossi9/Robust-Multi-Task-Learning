from DatasetGenerator import *
from Optimizer import *
from Optimizer_AMHT_LRS import *
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

# Proximal method__________________________________________________________________________________________________

optimizer = Optimizers(dataset["features"], dataset["labels"])
step_size = 5e-3
n_iter = 30000

low_rank_mat = np.zeros((high_dim, n_tasks))
sparse_mat = np.zeros((high_dim, n_tasks))

display_regularization_params(dataset)

results = optimizer.proximal_method(step_size, 0.2, 2.0, n_iter, low_rank_mat, sparse_mat,
                                  dataset["features"], dataset["labels"], dataset["features"], dataset["labels"])

# save the results
if save_res:
    path = "Results/plot-3/run_7_sparse.pkl"  # change accordingly depending on which test are we doing
    with open(path, 'wb') as handle:
        pickle.dump({"dataset": dataset, "results": results}, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("FINISHED OPTIMIZING")

# ANALYSIS RESULTS
loss_analysis(results["losses"], dataset, optimizer)
ratio_of_recovery_analysis(dataset, results)
matrices_info(results, dataset, sparsity, low_dim)


