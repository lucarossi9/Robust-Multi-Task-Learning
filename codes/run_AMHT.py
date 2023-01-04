from DatasetGenerator import *
from Optimizer_AMHT_LRS import *
from Utils import *
import seaborn as sns
sns.set_theme()

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

# set initial iterates
low_rank_mat = np.zeros((high_dim, n_tasks))
sparse_mat = np.zeros((high_dim, n_tasks))
U_0 = np.random.normal(size=(high_dim, low_dim), scale=1)
U_0, null = np.linalg.qr(U_0)
W_0 = np.zeros((low_dim, n_tasks))

features_train = dataset["features"]
features_test = dataset["features"]
labels_train = dataset["labels"]
labels_test = dataset["labels"]

gamma = 10  # 10
B = 10  # 10
n_iters = 50
c_3 = 0.5
c_4 = 1 / 50
c_5 = 1 / 50
lr = 0.005
optimizer_google = Optimizer_AMHT_LRS(dataset["features"], dataset["labels"])
results = optimizer_google.AMHT_LRS(n_iters, 7, gamma, B, U_0, W_0, sparse_mat, features_train, features_train,
                                    labels_train, labels_train, c_3, c_4, c_5, lr, eps=1e-2)

predicted_low_ranks = [results["Us"][i] @ results["Ws"][i] for i in range(len(results["Us"]))]
results["low_ranks"] = predicted_low_ranks

# ANALYSIS RESULTS
loss_analysis(results["losses"], dataset, optimizer_google)
ratio_of_recovery_analysis(dataset, results)
matrices_info(results, dataset, sparsity, low_dim)

