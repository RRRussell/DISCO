import numpy as np
from scipy.spatial import distance, distance_matrix
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import mean_squared_error, f1_score
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

        weights1 = np.ones(n1) / n1
        weights2 = np.ones(n2) / n2

        dist_matrix = distance_matrix(points1, points2)

        row_ind, col_ind = linear_sum_assignment(dist_matrix)

        emd_value = dist_matrix[row_ind, col_ind].sum() / max(n1, n2)

        return emd_value

    @staticmethod
    def evaluate_expression(true_coords, true_gene_expressions, pred_coords, pred_gene_expressions):
        dist_matrix = distance_matrix(pred_coords, true_coords)
        closest_indices = np.argmin(dist_matrix, axis=1)
        
        closest_true_gene_expressions = true_gene_expressions[closest_indices]
        
        mse = mean_squared_error(closest_true_gene_expressions, pred_gene_expressions)
        
        threshold = np.median(true_gene_expressions)
        true_binary = (closest_true_gene_expressions >= threshold).astype(int)
        pred_binary = (pred_gene_expressions >= threshold).astype(int)
        f1 = f1_score(true_binary, pred_binary, average='micro')
        
        cosine_sim = np.mean(cosine_similarity(closest_true_gene_expressions, pred_gene_expressions))
        
        return mse, f1, cosine_sim

















