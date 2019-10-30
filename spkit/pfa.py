from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
import copy

algorithm_lookup = {
    "sklearn": LogisticRegression,
    "sm": sm.GLM
}

default_params = {
    "sklearn": {},
    "sm": {"family": sm.families.Binomial()}
}

sm_regularized_params = ["alpha", "start_params", "refit", "L1_wt",
                         "cnvrg_tol", "maxiter", "zero_tol"]


class PFA(object):
    def __init__(self, lib="sklearn"):
        """ Init PFA object

        Parameters
        ----------
        lib: sklearn uses Scikit-Learn LogisticRegression module. sm uses
            StatsModel glm module.
        """
        # Lib to use
        self.lib = lib
        # Model
        self.model = None
        self.model_params = None
        # Params
        self.params = None
        self.skills = None
        
        # Student learning state
        self.learning_state = []
        
        # Evaluation metrics
        self.n_skills = 0
        self.n_questions = 0
        self.outcomes = None
        self.loglikelihood = 0
        self.outcome_prob = 0
        self.aic = 0
        self.bic = 0
        self.rmse = 0
        self.acc = 0

    @staticmethod
    def _create_onehot(row, skills, skills_onehot):
        """ Transform each row to its one hot version """
        idx = np.where(skills == row['skill'])[0]
        wins = row['wins']*skills_onehot[idx][0]
        fails = row['fails']*skills_onehot[idx][0]
        return np.concatenate((skills_onehot[idx][0], wins, fails))
        
    def _apply_onehot(self, data):
        """ Transform data to its onehot format """
        skills = self.skills
        skills_onehot = self.params["skills_onehot"]
        onehot_array = data.apply(self._create_onehot, axis=1,
                                  args=(skills, 
                                        skills_onehot))
        cols = ["skills_%d"%skill for skill in skills]
        cols += ["wins_%d"%skill for skill in skills]
        cols += ["fails_%d"%skill for skill in skills]
        onehot_df = pd.DataFrame(onehot_array.tolist(), columns=cols)
        data = pd.concat((data, onehot_df), axis=1)
        data = data.drop(columns=['skill', 'wins', 'fails'])
        data = data.groupby(['index']).sum().astype({
            'outcome': 'bool'}).astype({'outcome': 'int64'})
        return data, cols

    def _skills_onehot(self, data):
        """ Transform PFA data to its onehot version where each columns
        represents a skill, wins and fails for the corresponding skills. A
        table header for 2 skills would contain the following information
        skill_1 | skill_2 | wins_1 | wins_2 | fails_1 | fails_2
        """
        skills = data['skill'].unique()
        self.skills = skills.tolist()
        self.n_skills = len(self.skills)
        skills_array = skills.reshape(-1, 1)
        enc = OneHotEncoder(categories='auto', sparse=False)
        skills_onehot = enc.fit_transform(skills_array)
        self.params["skills_onehot"] = skills_onehot
        return skills_onehot
		
    def _transform_student_data(self, data, q_matrix, learning_state=None):
        """ Transform original data into PFA expected format. Calculates wins,
        fails, get skills and transform everything into one-hot variables """
        student_skills_count = {}
        pfa_data = []
        for idx, row in enumerate(data):
            outcome, question_id = row

            # Get skill index in skills list
            skills_idx = np.where(q_matrix[question_id, :] == 1)
            # For each skill, calculate wins and fails
            for skill in skills_idx[0]:
                # Create skills for student if it's new
                if skill not in student_skills_count:
                    student_skills_count[skill] = defaultdict(int)

                # Add row to PFA table
                wins = student_skills_count[skill]["wins"]
                fails = student_skills_count[skill]["fails"]
                pfa_row = (idx, skill, wins, fails, outcome)
                pfa_data.append(pfa_row)

                # Update wins or fails counter
                if outcome == 1:
                    student_skills_count[skill]["wins"] += 1
                else:
                    student_skills_count[skill]["fails"] += 1

        # Create dataframe from PFA data
        df = pd.DataFrame(pfa_data, columns=["index", "skill", "wins",
                                             "fails", "outcome"])
        pfa_onehot = self._apply_onehot(df)
        # Sum learning state (previous wins and fails)
        if learning_state:
            for idx, skill in enumerate(self.skills):
                pfa_onehot["wins_%s" % skill] += learning_state[idx][0]
                pfa_onehot["fails_%s" % skill] += learning_state[idx][1]
        return pfa_onehot

    def _transform_data(self, data, q_matrix):
        """ Transform original data into PFA expected format. Calculates wins,
        fails, get skills and transform everything into one-hot variables """
        skills_count = {}
        pfa_data = []
        for idx, row in enumerate(data):
            outcome, student_id, question_id = row
            # If student was not seen yet, create student counter
            if student_id not in skills_count:
                skills_count[student_id] = {}

            # Get skill index in skills list
            skills_idx = np.where(q_matrix[question_id, :] == 1)
            # For each skill, calculate wins and fails
            for skill in skills_idx[0]:
                # Create skills for student if it's new
                if skill not in skills_count[student_id]:
                    skills_count[student_id][skill] = defaultdict(int)

                # Add row to PFA table
                wins = skills_count[student_id][skill]["wins"]
                fails = skills_count[student_id][skill]["fails"]
                pfa_row = (idx, skill, wins, fails, outcome)
                pfa_data.append(pfa_row)

                # Update wins or fails counter
                if outcome == 1:
                    skills_count[student_id][skill]["wins"] += 1
                else:
                    skills_count[student_id][skill]["fails"] += 1

        # Create dataframe from PFA data
        df = pd.DataFrame(pfa_data, columns=["index", "skill", "wins", "fails",
                                             "outcome"])
        self._skills_onehot(df)
        pfa_onehot = self._apply_onehot(df)
        return pfa_onehot

    def fit(self, data, q_matrix, **kwargs):
        """ Fit PFA model to data.


        Parameters
        ----------
        data : {array-like}, shape (n_steps, 3)
            Sequence of students steps. Each of the three dimensions are:
            Observed outcome: 0 for fail and 1 for success
            Student id: student unique identifier
            Question id: question id in q_matrix

        q_matrix: matrix, shape (n_questions, n_concepts)
            Each row is a question and each column a concept.
            If the concept is present in the question, the
            correspondent cell should contain 1, otherwise, 0.

        kwargs = extra parameters to be passed to sklearn or sm libraries. If
            penalty parameter is set to sm library, fit_regularized method is
            used.

        Returns
        -------
        self: object

        Notes
        -----
        To obtain close results using regularization in sklearn and sm
        libraries, use the following equalities between parameters:
        (sklearn) C = 1/(n * alpha) (sm)
        (sklearn) l1_ratio = L1_wt (sm)

        Results may still be different due to different solver algorithms.

        """
        # Transform data to PFA format
        self.params = {}
        data, cols = self._transform_data(data, q_matrix)

        # Fit model
        X = data[cols]
        y = data['outcome']

        # Fit data
        params = copy.deepcopy(default_params[self.lib])
        params.update(kwargs)
        self.model_params = params

        if self.lib == "sklearn":
            model = algorithm_lookup[self.lib](**params)
            model.fit(X, y)
            self.params["weights"] = np.concatenate((model.intercept_.reshape(-1,1),
                                          model.coef_), axis=1)
        elif self.lib == "sm":
            # If sm library, it is needed to add the intercept
            X = sm.add_constant(X)
            model = algorithm_lookup[self.lib](y, X, **params)
            if "penalty" in params:
                reg_params = {}
                for key in sm_regularized_params:
                    try:
                        reg_params[key] = params[key]
                    except KeyError:
                        pass
                model = model.fit_regularized(**reg_params)
            else:
                model = model.fit()
            
            self.params["weights"] = model.params.values.reshape(1,-1)

        self.model = model
        return self

    def _predict(self, data, q_matrix, learning_state=None):
        """ Predict student outcomes based on trained model.

        Parameters
        ----------
        data : {array-like}, shape (n_steps, 2)
            Sequence of student steps. Each of the three dimensions are:
            Observed outcome: 0 for fail and 1 for success
            Question id: question id in q_matrix

        q_matrix: matrix, shape (n_questions, n_concepts)
            Each row is a question and each column a concept.
            If the concept is present in the question, the
            correspondent cell should contain 1, otherwise, 0.

        Returns
        -------
        outcome : {array-like}, shape (n_steps, 2)
            Outcome probabilites for steps in data. Column 0 corresponds to
            outcome 0 (incorrect) and column 1 to outcome 1 (correct)
        """
        # Transform data to PFA format
        data, cols = self._transform_student_data(data, q_matrix, learning_state)

        # Fit model
        X = data[cols]
        self.outcomes = data['outcome']
        self.n_questions = self.outcomes.shape[0]
        
        if self.lib == "sklearn":
            self.outcome_prob = self.model.predict_proba(X)
        elif self.lib == "sm":
            # If sm library, it is needed to add the intercept
            X = sm.add_constant(X)
            self.outcome_prob = np.zeros((X.shape[0], 2))
            self.outcome_prob[:,1] = self.model.predict(X).values
            self.outcome_prob[:,0] = 1 - self.outcome_prob[:,1]
            
        # Update student learning state
        for skill in self.skills:
            worked_skill = X[X['skills_%s' % skill] == 1]
            if not worked_skill.empty: 
                wins = X[X['skills_%s' % skill] == 1]['wins_%s' % skill].tail(1).iloc[0]
                fails = X[X['skills_%s' % skill] == 1]['fails_%s' % skill].tail(1).iloc[0]
            else:
                wins = 0
                fails = 0
    
        return self.outcome_prob

    def predict_proba(self, data, q_matrix, learning_state=[]):
        """ Predict student outcome probabilities based on trained model.

        Parameters
        ----------
        data : {array-like}, shape (n_steps, 2)
            Sequence of student steps. Each of the two dimensions are:
            Observed outcome: 0 for fail and 1 for success
            Question id: question id in q_matrix

        q_matrix: matrix, shape (n_questions, n_concepts)
            Each row is a question and each column a concept.
            If the concept is present in the question, the
            correspondent cell should contain 1, otherwise, 0.

        Returns
        -------
        outcome : {array-like}, shape (n_steps, 2)
            Outcome probabilites for steps in data. Column 0 corresponds to
            outcome 0 (incorrect) and column 1 to outcome 1 (correct)

        """
        y_pred_proba = self._predict(data, q_matrix, learning_state)
        return y_pred_proba

    def predict(self, data, q_matrix, learning_state=[]):
        """ Predict student outcomes based on trained model. This is just the
        hard-assigment (highest probability) of the outcome probabilities.

        Parameters
        ----------
        data : {array-like}, shape (n_steps, 2)
            Sequence of student steps. Each of the two dimensions are:
            Observed outcome: 0 for fail and 1 for success
            Question id: question id in q_matrix

        q_matrix: matrix, shape (n_questions, n_concepts)
            Each row is a question and each column a concept.
            If the concept is present in the question, the
            correspondent cell should contain 1, otherwise, 0.

        Returns
        -------
        outcome : {array-like}, shape (n_steps, 2)
            Outcome probabilites for steps in data. Column 0 corresponds to
            outcome 0 (incorrect) and column 1 to outcome 1 (correct)
        """
        y_pred_proba = self._predict(data, q_matrix, learning_state)
        y_pred = np.argmax(y_pred_proba, axis=1)
        return y_pred

    def score(self):
        """
        Calculates LL, AIC, BIC, RMSE and accuracy for the predicted sample

        """
        y = self.outcomes
        
        # LL
        self.loglikelihood = (np.log(self.outcome_prob[range(y.shape[0]),y]))

        # AIC: 7 is the number of PFA parameters
        self.aic = -2*self.loglikelihood.sum() + 2*3*self.n_skills

        # BIC: 7 is the number of PFA parameters
        self.bic = -2*self.loglikelihood.sum() + 3*self.n_skills*np.log(y.shape[0])

        # RMSE
        rmse = ((1-self.outcome_prob[range(y.shape[0]), y])**2).sum()
        self.rmse = np.sqrt(rmse/y.shape[0])

        # Accuracy
        estimated_outcome = np.argmax(self.outcome_prob, axis=1)
        self.acc = (estimated_outcome == y).sum()/y.shape[0]

        return self.loglikelihood.sum(), self.aic, self.bic, self.rmse, self.acc

    def get_params(self):
        """ Get fitted params.

        Returns
        -------
        params : list. List containing the prior, transition and emission values for each skill.
        """
        if self.params is None:
            raise RuntimeError("You should run fit before getting params")
        return self.params

    def set_params(self, params):
        """ Set model params. No validation is done for this function.
        Make sure the params variable is in the expected format.

        Returns
        -------
        self: object
        """
        self.params = params
        return self
       
    def get_learning_state(self):
        """ Return last predict student learning state.

        Returns
        -------
        learning_state: array. Array with last predicted student learning state value for each KC.
        """
        if self.learning_state is None:
            raise RuntimeError("You should predict outcomes for a student before getting the learning state")
        return self.learning_state