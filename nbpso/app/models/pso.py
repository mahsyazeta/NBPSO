import numpy as np
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer

class NaiveBayes:
    def __init__(self, model_type):
        if model_type == 'multinomial':
            self.clf = MultinomialNB()
        elif model_type == 'bernoulli':
            self.clf = BernoulliNB()
        elif model_type == 'gaussian':
            self.clf = GaussianNB()
        else:
            raise ValueError("Invalid model type. Choose 'multinomial', 'bernoulli', or 'gaussian'.")

    def fit(self, X, y, selected_features):
        if isinstance(self.clf, GaussianNB):
            self.clf.fit(X[:, selected_features].toarray(), y)
        else:
            self.clf.fit(X[:, selected_features], y)

def evaluate_features(X_train, y_train, selected_features, classifier):
    kfold = 10
    clf = classifier.clf
    if isinstance(clf, GaussianNB):
        X_selected = X_train[:, selected_features].toarray()
    else:
        X_selected = X_train[:, selected_features]
    cv_results = cross_val_score(clf, X_selected, y_train, cv=kfold, scoring='accuracy')
    return cv_results.mean()

def pso_feature_selection(X_train, y_train, model_type, n_particles=20, inertia=0.8, global_weight=1, local_weight=0.9, tol=1e-5, patience=10):
    num_samples, num_features = X_train.shape
    bounds = [0, 1]
    num_particles = n_particles
    dimensions = num_features
    particles = np.random.rand(num_particles, dimensions)
    velocities = np.random.rand(num_particles, dimensions) * 0.1
    best_positions = particles.copy()
    best_scores = np.zeros(num_particles)

    global_best_position = np.zeros(dimensions)
    global_best_score = 0

    no_improvement_count = 0
    previous_global_best_score = 0

    while no_improvement_count < patience:
        for particle in range(num_particles):
            r1 = np.random.rand(dimensions)
            r2 = np.random.rand(dimensions)
            velocities[particle] = (inertia * velocities[particle] +
                                    global_weight * r1 * (best_positions[particle] - particles[particle]) +
                                    local_weight * r2 * (global_best_position - particles[particle]))

            particles[particle] += velocities[particle]
            particles[particle] = np.clip(particles[particle], bounds[0], bounds[1])

            selected_features = particles[particle] > 0.5
            nb = NaiveBayes(model_type)
            nb.fit(X_train, y_train, selected_features)
            accuracy = evaluate_features(X_train, y_train, selected_features, nb)

            if accuracy > best_scores[particle]:
                best_scores[particle] = accuracy
                best_positions[particle] = particles[particle].copy()

            if accuracy > global_best_score:
                global_best_score = accuracy
                global_best_position = particles[particle].copy()

        if abs(global_best_score - previous_global_best_score) < tol:
            no_improvement_count += 1
        else:
            no_improvement_count = 0
        
        previous_global_best_score = global_best_score

    return global_best_position > 0.5

def optimize_feature_selection(df, model_type):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['Text Tweet'])
    y = df['Sentiment']

    best_features = pso_feature_selection(X, y, model_type)

    selected_indices = [i for i, selected in enumerate(best_features) if selected]
    best_params = {'selected_indices': selected_indices}
    return best_params
