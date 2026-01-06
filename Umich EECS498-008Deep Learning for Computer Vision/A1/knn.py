"""
Implements a K-Nearest Neighbor classifier in PyTorch.
"""
import chunk
import statistics
from matplotlib import axes, axis
import torch
from typing import Dict, List


def hello():
    """
    This is a sample function that we will try to import and run to ensure that
    our environment is correctly set up on Google Colab.
    """
    print("Hello from knn.py!")


def compute_distances_two_loops(x_train: torch.Tensor, x_test: torch.Tensor):
    """
    Computes the squared Euclidean distance between each element of training
    set and each element of test set. Images should be flattened and treated
    as vectors.

    This implementation uses a naive set of nested loops over the training and
    test data.

    The input data may have any number of dimensions -- for example this
    function should be able to compute nearest neighbor between vectors, in
    which case the inputs will have shape (num_{train, test}, D); it should
    also be able to compute nearest neighbors between images, where the inputs
    will have shape (num_{train, test}, C, H, W). More generally, the inputs
    will have shape (num_{train, test}, D1, D2, ..., Dn); you should flatten
    each element of shape (D1, D2, ..., Dn) into a vector of shape
    (D1 * D2 * ... * Dn) before computing distances.

    The input tensors should not be modified.

    NOTE: Your implementation may not use `torch.norm`, `torch.dist`,
    `torch.cdist`, or their instance method variants (`x.norm`, `x.dist`,
    `x.cdist`, etc.). You may not use any functions from `torch.nn` or
    `torch.nn.functional` modules.

    Args:
        x_train: Tensor of shape (num_train, D1, D2, ...)
        x_test: Tensor of shape (num_test, D1, D2, ...)

    Returns:
        dists: Tensor of shape (num_train, num_test) where dists[i, j]
            is the squared Euclidean distance between the i-th training point
            and the j-th test point. It should have the same dtype as x_train.
    """
    # Initialize dists to be a tensor of shape (num_train, num_test) with the
    # same datatype and device as x_train
    num_train = x_train.shape[0]
    num_test = x_test.shape[0]
    dists = x_train.new_zeros(num_train, num_test)
    ##########################################################################
    # TODO: Implement this function using a pair of nested loops over the    #
    # training data and the test data.                                       #
    #                                                                        #
    # You may not use torch.norm (or its instance method variant), nor any   #
    # functions from torch.nn or torch.nn.functional.                        #
    ##########################################################################
    # Replace "pass" statement with your code
    shaped_train = x_train.view(num_train, -1)
    shaped_test = x_test.view(num_test, -1)
    for i in range(num_train):
        for j in range(num_test):
            dists[i, j] = ((shaped_train[i] - shaped_test[j]) *
                           (shaped_train[i] - shaped_test[j])).sum()
    ##########################################################################
    #                           END OF YOUR CODE                             #
    ##########################################################################
    return dists


def compute_distances_one_loop(x_train: torch.Tensor, x_test: torch.Tensor):
    """
    Computes the squared Euclidean distance between each element of training
    set and each element of test set. Images should be flattened and treated
    as vectors.

    This implementation uses only a single loop over the training data.

    Similar to `compute_distances_two_loops`, this should be able to handle
    inputs with any number of dimensions. The inputs should not be modified.

    NOTE: Your implementation may not use `torch.norm`, `torch.dist`,
    `torch.cdist`, or their instance method variants (`x.norm`, `x.dist`,
    `x.cdist`, etc.). You may not use any functions from `torch.nn` or
    `torch.nn.functional` modules.

    Args:
        x_train: Tensor of shape (num_train, D1, D2, ...)
        x_test: Tensor of shape (num_test, D1, D2, ...)

    Returns:
        dists: Tensor of shape (num_train, num_test) where dists[i, j]
            is the squared Euclidean distance between the i-th training point
            and the j-th test point. It should have the same dtype as x_train.
    """
    # Initialize dists to be a tensor of shape (num_train, num_test) with the
    # same datatype and device as x_train
    num_train = x_train.shape[0]
    num_test = x_test.shape[0]
    dists = x_train.new_zeros(num_train, num_test)
    ##########################################################################
    # TODO: Implement this function using only a single loop over x_train.   #
    #                                                                        #
    # You may not use torch.norm (or its instance method variant), nor any   #
    # functions from torch.nn or torch.nn.functional.                        #
    ##########################################################################
    # Replace "pass" statement with your code
    x_train = x_train.reshape(num_train, -1)
    x_test = x_test.reshape(num_test, -1)

    for i in range(num_train):
        dists[i] = (((x_test - x_train[i]) * (x_test - x_train[i])).sum(dim=1))

    ##########################################################################
    #                           END OF YOUR CODE                             #
    ##########################################################################
    return dists


'''
def compute_distances_no_loops(x_train: torch.Tensor, x_test: torch.Tensor):
    """
    Computes the squared Euclidean distance between each element of training
    set and each element of test set. Images should be flattened and treated
    as vectors.

    This implementation should not use any Python loops. For memory-efficiency,
    it also should not create any large intermediate tensors; in particular you
    should not create any intermediate tensors with O(num_train * num_test)
    elements.

    Similar to `compute_distances_two_loops`, this should be able to handle
    inputs with any number of dimensions. The inputs should not be modified.

    NOTE: Your implementation may not use `torch.norm`, `torch.dist`,
    `torch.cdist`, or their instance method variants (`x.norm`, `x.dist`,
    `x.cdist`, etc.). You may not use any functions from `torch.nn` or
    `torch.nn.functional` modules.

    Args:
        x_train: Tensor of shape (num_train, C, H, W)
        x_test: Tensor of shape (num_test, C, H, W)

    Returns:
        dists: Tensor of shape (num_train, num_test) where dists[i, j] is
            the squared Euclidean distance between the i-th training point and
            the j-th test point.
    """
    # Initialize dists to be a tensor of shape (num_train, num_test) with the
    # same datatype and device as x_train
    num_train = x_train.shape[0]
    num_test = x_test.shape[0]
    dists = x_train.new_zeros(num_train, num_test)
    ##########################################################################
    # TODO: Implement this function without using any explicit loops and     #
    # without creating any intermediate tensors with O(num_train * num_test) #
    # elements.                                                              #
    #                                                                        #
    # You may not use torch.norm (or its instance method variant), nor any   #
    # functions from torch.nn or torch.nn.functional.                        #
    #                                                                        #
    # HINT: Try to formulate the Euclidean distance using two broadcast sums #
    #       and a matrix multiply.                                           #
    ##########################################################################
    # Replace "pass" statement with your code
    x_train = x_train.reshape(num_train, -1)
    x_test = x_test.reshape(num_test, -1)
    x_train = x_train.reshape(x_train.shape[0], 1, -1)
    dists = ((x_train - x_test) * (x_train - x_test)).sum(dim=2)
    ##########################################################################
    #                           END OF YOUR CODE                             #
    ##########################################################################
    return dists

'''


# ||a - b||^2 = ||a||^2 + ||b||^2 - 2 * a·b
def compute_distances_no_loops(x_train: torch.Tensor,
                               x_test: torch.Tensor) -> torch.Tensor:
    """
    Fully vectorized implementation – **no Python loops** and **no
    O(N·M) intermediate tensors** (apart from the final output).

    Uses the identity a·b = ab.t()
        ||a - b||^2 = ||a||^2 + ||b||^2 - 2 * a·b
    which only needs two O(N) / O(M) vectors and one matrix multiply.
    """
    # --------------------------------------------------------------
    # 1️⃣ 形状统一 → (N, D) / (M, D)   (flatten all but the first dim)
    # --------------------------------------------------------------
    N = x_train.shape[0]
    M = x_test.shape[0]

    train_flat = x_train.view(N, -1)          # (N, D)
    test_flat = x_test.view(M, -1)           # (M, D)

    # --------------------------------------------------------------
    # 2️⃣ 计算每个向量的平方范数（长度为 N、M 的向量）
    #    注意：如果输入是整数类型，直接求平方会溢出 → 先 cast 到 float
    # --------------------------------------------------------------
    dtype = train_flat.dtype
    if not torch.is_floating_point(train_flat):
        # 转成 float32（或 float64，取决于原始 dtype 的位宽）
        train_flat = train_flat.to(torch.float32)
        test_flat = test_flat.to(torch.float32)

    # a_norm_sq[i] = sum_k train_flat[i,k]^2
    a_norm_sq = (train_flat * train_flat).sum(dim=1)   # (N,)

    # b_norm_sq[j] = sum_k test_flat[j,k]^2
    b_norm_sq = (test_flat * test_flat).sum(dim=1)    # (M,)

    # --------------------------------------------------------------
    # 3️⃣ 矩阵乘法  a @ b.T   → (N, M)   (high‑performance BLAS)
    # --------------------------------------------------------------
    # 对于大规模数据，`torch.matmul` 会自动使用 cuBLAS / MKL 等库，
    # 并且会在内部做块式（blocked）计算，不会一次性分配额外的 (N,M,D) 张量。
    cross_term = torch.matmul(train_flat, test_flat.t())   # 内积: (N, M)
    # 2
    # --------------------------------------------------------------
    # 4️⃣ 通过广播把标量范数加到矩阵上
    #    a_norm_sq[:, None] → (N,1)   与 (N,M) 广播
    #    b_norm_sq[None, :] → (1,M)   与 (N,M) 广播
    # --------------------------------------------------------------
    dists = a_norm_sq[:, None] + b_norm_sq[None, :] - 2.0 * cross_term

    # 由于浮点数误差，可能出现极小的负数 → clamp 为 0
    dists.clamp_min_(0)

    # 最终返回的 dtype / device 必须和 x_train 完全一致
    if dists.dtype != x_train.dtype:
        dists = dists.to(x_train.dtype)
    return dists


def predict_labels(dists: torch.Tensor, y_train: torch.Tensor, k: int = 1):
    """
    Given distances between all pairs of training and test samples, predict a
    label for each test sample by taking a MAJORITY VOTE among its `k` nearest
    neighbors in the training set.

    In the event of a tie, this function SHOULD return the smallest label. For
    example, if k=5 and the 5 nearest neighbors to a test example have labels
    [1, 2, 1, 2, 3] then there is a tie between 1 and 2 (each have 2 votes),
    so we should return 1 since it is the smallest label.

    This function should not modify any of its inputs.

    Args:
        dists: Tensor of shape (num_train, num_test) where dists[i, j] is the
            squared Euclidean distance between the i-th training point and the
            j-th test point.
        y_train: Tensor of shape (num_train,) giving labels for all training
            samples. Each label is an integer in the range [0, num_classes - 1]
        k: The number of nearest neighbors to use for classification.

    Returns:
        y_pred: int64 Tensor of shape (num_test,) giving predicted labels for
            the test data, where y_pred[j] is the predicted label for the j-th
            test example. Each label should be an integer in the range
            [0, num_classes - 1].
    """
    num_train, num_test = dists.shape
    y_pred = torch.zeros(num_test, dtype=torch.int64)
    ##########################################################################
    # TODO: Implement this function. You may use an explicit loop over the   #
    # test samples.                                                          #
    #                                                                        #
    # HINT: Look up the function torch.topk                                  #
    ##########################################################################
    # Replace "pass" statement with your code
    # _ : 距离值（我们不需要），topk_idx : 对应的训练样本下标，形状 (k, num_test)
    _, topk_idx = torch.topk(dists,
                             k,              # 取k个值
                             dim=0,          # 按列（即每个 test 样本）找 top‑k, 返回dim=0的标
                             largest=False,   # 取最小的距离
                             sorted=True)     # 让返回的下标已经按距离从小到大排好序

    # --------------------------------------------------------------
    # 2️⃣ 计算标签的种类数（后面用 torch.bincount 需要知道长度）
    # --------------------------------------------------------------
    num_classes = int(y_train.max().item()) + 1   # 标签范围是 [0, C‑1]

    # --------------------------------------------------------------
    # 3️⃣ 对每个测试样本做多数投票
    #    - 取出 k 条最近邻的标签
    #    - 用 torch.bincount 统计每个标签出现的次数
    #    - 用 torch.argmax 取出现次数最多的标签
    #      （在出现相同最大次数时，argmax 会返回第一个出现的下标，
    #       因为 bincount 的结果是按标签值顺序排列的，这恰好实现
    #       “返回最小标签”的需求）
    # --------------------------------------------------------------

    for j in range(num_test):
        # 最近邻的训练样本下标（长度为 k）
        neighbour_idx = topk_idx[:, j]               # shape (k,)
        # 对应的标签
        neighbour_labels = y_train[neighbour_idx]    # shape (k,)

        # 统计每个标签出现的次数，长度保证为 num_classes
        votes = torch.bincount(neighbour_labels,
                               minlength=num_classes)  # shape (num_classes,)
        # 返回vector

        # 取出现次数最多的标签；若出现平局，返回最小的标签
        y_pred[j] = torch.argmax(votes)
    ##########################################################################
    #                           END OF YOUR CODE                             #
    ##########################################################################
    return y_pred


class KnnClassifier:

    def __init__(self, x_train: torch.Tensor, y_train: torch.Tensor):
        """
        Create a new K-Nearest Neighbor classifier with the specified training
        data. In the initializer we simply memorize the provided training data.

        Args:
            x_train: Tensor of shape (num_train, C, H, W) giving training data
            y_train: int64 Tensor of shape (num_train, ) giving training labels
        """
        ######################################################################
        # TODO: Implement the initializer for this class. It should perform  #
        # no computation and simply memorize the training data in            #
        # `self.x_train` and `self.y_train`, accordingly.                    #
        ######################################################################
        # Replace "pass" statement with your code
        self.x_train = x_train
        self.y_train = y_train
        ######################################################################
        #                         END OF YOUR CODE                           #
        ######################################################################

    def predict(self, x_test: torch.Tensor, k: int = 1):
        """
        Make predictions using the classifier.

        Args:
            x_test: Tensor of shape (num_test, C, H, W) giving test samples.
            k: The number of neighbors to use for predictions.

        Returns:
            y_test_pred: Tensor of shape (num_test,) giving predicted labels
                for the test samples.
        """
        y_test_pred = None
        ######################################################################
        # TODO: Implement this method. You should use the functions you      #
        # wrote above for computing distances (use the no-loop variant) and  #
        # to predict output labels.                                          #
        ######################################################################
        # Replace "pass" statement with your code
        dist = compute_distances_no_loops(self.x_train, x_test)
        y_test_pred = predict_labels(dist, self.y_train, k)
        ######################################################################
        #                         END OF YOUR CODE                           #
        ######################################################################
        return y_test_pred

    def check_accuracy(
        self,
        x_test: torch.Tensor,
        y_test: torch.Tensor,
        k: int = 1,
        quiet: bool = False
    ):
        """
        Utility method for checking the accuracy of this classifier on test
        data. Returns the accuracy of the classifier on the test data, and
        also prints a message giving the accuracy.

        Args:
            x_test: Tensor of shape (num_test, C, H, W) giving test samples.
            y_test: int64 Tensor of shape (num_test,) giving test labels.
            k: The number of neighbors to use for prediction.
            quiet: If True, don't print a message.

        Returns:
            accuracy: Accuracy of this classifier on the test data, as a
                percent. Python float in the range [0, 100]
        """
        y_test_pred = self.predict(x_test, k=k)
        num_samples = x_test.shape[0]
        num_correct = (y_test == y_test_pred).sum().item()
        accuracy = 100.0 * num_correct / num_samples
        msg = (
            f"Got {num_correct} / {num_samples} correct; "
            f"accuracy is {accuracy:.2f}%"
        )
        if not quiet:
            print(msg)
        return accuracy


def knn_cross_validate(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    num_folds: int = 5,
    k_choices: List[int] = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100],
):
    """
    Perform cross-validation for `KnnClassifier`.

    Args:
        x_train: Tensor of shape (num_train, C, H, W) giving all training data.
        y_train: int64 Tensor of shape (num_train,) giving labels for training
            data.
        num_folds: Integer giving the number of folds to use.
        k_choices: List of integers giving the values of k to try.

    Returns:  {k:List[i-th fold]}
        k_to_accuracies: Dictionary mapping values of k to lists, where
            k_to_accuracies[k][i] is the accuracy on the i-th fold of a
            `KnnClassifier` that uses k nearest neighbors.
    """

    # First we divide the training data into num_folds equally-sized folds.
    x_train_folds = []
    y_train_folds = []
    ##########################################################################
    # TODO: Split the training data and images into folds. After splitting,  #
    # x_train_folds and y_train_folds should be lists of length num_folds,   #
    # where y_train_folds[i] is label vector for images inx_train_folds[i].  #
    #                                                                        #
    # HINT: torch.chunk                                                      #
    ##########################################################################
    # Replace "pass" statement with your code
    x_train = x_train.reshape(x_train.shape[0], -1)
    x_train_folds = list(torch.chunk(x_train, num_folds))
    y_train_folds = list(torch.chunk(y_train, num_folds))

    # results
    k_to_accuracies: Dict[int, List[float]] = {k: [] for k in k_choices}

    for fold_idx in range(num_folds):
        # validation set
        x_validation = x_train_folds[fold_idx]
        y_validation = y_train_folds[fold_idx]

        # train set
        x_train_parts = x_train_folds[:fold_idx] + x_train_folds[fold_idx + 1:]
        y_train_parts = y_train_folds[:fold_idx] + y_train_folds[fold_idx + 1:]

        x_train_concat = torch.cat(x_train_parts, dim=0)
        y_train_concat = torch.cat(y_train_parts, dim=0)

        # init knn class
        cls = KnnClassifier(torch.tensor(x_train_concat),
                            torch.tensor(y_train_concat))

        for k in k_choices:
            k_to_accuracies[k].append(cls.check_accuracy(
                torch.tensor(x_validation), torch.tensor(y_validation), k, False))

    ##########################################################################
    #                           END OF YOUR CODE                             #
    ##########################################################################

    # A dictionary holding the accuracies for different values of k that we
    # find when running cross-validation. After running cross-validation,
    # k_to_accuracies[k] should be a list of length num_folds giving the
    # different accuracies we found trying `KnnClassifier`s using k neighbors.
    # k_to_accuracies = {}

    ##########################################################################
    # TODO: Perform cross-validation to find the best value of k. For each   #
    # value of k in k_choices, run the k-NN algorithm `num_folds` times; in  #
    # each case you'll use all but one fold as training data, and use the    #
    # last fold as a validation set. Store the accuracies for all folds and  #
    # all values in k in k_to_accuracies.                                    #
    #                                                                        #
    # HINT: torch.cat                                                        #
    ##########################################################################
    # Replace "pass" statement with your code

    ##########################################################################
    #                           END OF YOUR CODE                             #
    ##########################################################################

    return k_to_accuracies


def knn_get_best_k(k_to_accuracies: Dict[int, List]):
    """
    Select the best value for k, from the cross-validation result from
    knn_cross_validate. If there are multiple k's available, then you SHOULD
    choose the smallest k among all possible answer.

    Args:
        k_to_accuracies: Dictionary mapping values of k to lists, where
            k_to_accuracies[k][i] is the accuracy on the i-th fold of a
            `KnnClassifier` that uses k nearest neighbors.

    Returns:
        best_k: best (and smallest if there is a conflict) k value based on
            the k_to_accuracies info.
    """
    best_k = 0
    ##########################################################################
    # TODO: Use the results of cross-validation stored in k_to_accuracies to #
    # choose the value of k, and store result in `best_k`. You should choose #
    # the value of k that has the highest mean accuracy accross all folds.   #
    ##########################################################################
    # Replace "pass" statement with your code
    max_accuracy = 0
    # may sorted the dict if there is a tie on accuracy
    # k_to_accuracies = dict(sorted(k_to_accuracies).item)
    for k, accuracys in k_to_accuracies.items():
        accuracy = statistics.mean(accuracys)
        if accuracy > max_accuracy:
            max_accuracy = accuracy
            best_k = k
    ##########################################################################
    #                           END OF YOUR CODE                             #
    ##########################################################################
    return best_k
