import numpy as np
from scipy.spatial import distance, distance_matrix
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity

class Evaluator:
    @staticmethod
    def chamfer_distance(real_points, generated_points):
        real_to_generated = np.mean(np.min(distance.cdist(real_points, generated_points, 'euclidean'), axis=1))
        generated_to_real = np.mean(np.min(distance.cdist(generated_points, real_points, 'euclidean'), axis=1))
        return real_to_generated + generated_to_real

    @staticmethod
    def calculate_emd(points1, points2):
        n1 = len(points1)
        n2 = len(points2)
        dist_matrix = distance_matrix(points1, points2)
        row_ind, col_ind = linear_sum_assignment(dist_matrix)
        emd_value = dist_matrix[row_ind, col_ind].sum() / max(n1, n2)
        return emd_value

    @staticmethod
    def evaluate_expression(true_coords, true_gene_expressions, pred_coords, pred_gene_expressions):
        # Compute the distance matrix between predicted and true coordinates
        dist_matrix = distance_matrix(pred_coords, true_coords)
        # Find the index of the closest true point for each predicted point
        closest_indices = np.argmin(dist_matrix, axis=1)
        # Get the closest true gene expressions
        closest_true_gene_expressions = true_gene_expressions[closest_indices]
        # Calculate mean squared error (MSE)
        mse = mean_squared_error(closest_true_gene_expressions, pred_gene_expressions)
        # Calculate L1 distance (mean absolute error)
        l1_distance = np.mean(np.abs(closest_true_gene_expressions - pred_gene_expressions))
        # Calculate cosine similarity
        cosine_sim = np.mean([cosine_similarity(
                                    closest_true_gene_expressions[i].reshape(1, -1),
                                    pred_gene_expressions[i].reshape(1, -1)
                                )[0, 0] for i in range(len(pred_gene_expressions))])
        return mse, l1_distance, cosine_sim

















