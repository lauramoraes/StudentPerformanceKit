from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
import copy
from multiprocessing import Pool, cpu_count
#from psutil import virtual_memory
from math import ceil

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

# Found this number by testing with different options. This number of cells per DF (rows x columns)
# when processing in parallel is the one that provided me the fastest processing.
# Feel free to play around with it.
MAX_CELLS = 825000

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
    def _create_onehot(row, skills, skills_onehot, cols=["wins", "fails"]):
        """ Transform each row to its one hot version """
        idx = np.where(skills == row['skill'])[0]
        new_row = skills_onehot[idx][0]
        for col in cols:
            onehot_col = row[col]*skills_onehot[idx][0]
            new_row = np.concatenate((new_row, onehot_col))
        return new_row
    
    @staticmethod
    def _change_type(data, col):
        mx = data[col].max()
        mn = data[col].min()
        if mn >= 0:
            if mx < 255:
                data[col] = data[col].astype(np.uint8)
            elif mx < 65535:
                data[col] = data[col].astype(np.uint16)
            elif mx < 4294967295:
                data[col] = data[col].astype(np.uint32)
            else:
                data[col] = data[col].astype(np.uint64)
        else:
            if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                data[col] = data[col].astype(np.int8)
            elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                data[col] = data[col].astype(np.int16)
            elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                data[col] = data[col].astype(np.int32)
            elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                data[col] = data[col].astype(np.int64)
        return data
        
    @staticmethod
    def _sum_df(data):
        return data[1].sum()
    
    def _get_number_of_splits(self, df, cpus):
        #available_mem = virtual_memory().available
        #one_row_mem = df.iloc[0].memory_usage()*len(self.onehot_cols)
        #rows_per_split = available_mem * 0.8 / (cpus * one_row_mem)
        #splits = ceil(df.shape[0]/rows_per_split)
        
        #if splits < cpus:
        #    return min(cpus, df.shape[0])
        #else:
        #    return splits
        rows_per_split = MAX_CELLS/len(self.onehot_cols)
        splits = ceil(df.shape[0]/rows_per_split)
        return splits
    
    def _create_onehot_cols(self, cols=["wins", "fails"]):
        onehot_cols = []
        for col in cols:
            onehot_cols += ["%s_%d" % (col, skill) for skill in self.skills]
        return onehot_cols
        
    def _apply_onehot(self, data):
        """ Transform data to its onehot format """
        #print("Applying to %d rows" % data.shape[0])
        skills = self.skills
        skills_onehot = self.params["skills_onehot"]
        onehot_array = data.apply(self._create_onehot, axis=1,
                                  args=(skills, skills_onehot, self.cols))
        # print("Onehot created")
        data = data.drop(columns=['skill'] + self.cols)
        # print("Cols dropped")
        data = data.reset_index(drop=True)
        # print("Index reseted")
        onehot_df = pd.DataFrame(onehot_array.tolist(), columns=self.onehot_cols, 
                                 dtype=np.uint16)
        # print("Extended df")
        data = pd.concat((data, onehot_df), axis=1)
        # print("Concat df")
        data = self._change_type(data, 'index')
        # print("DONE")
        return data
        
    def _onehot_encoder(self, data, col):
        """ Transform PFA data to its onehot version where each columns
        represents a skill, wins and fails for the corresponding skills. A
        table header for 2 skills would contain the following information
        skill_1 | skill_2 | wins_1 | wins_2 | fails_1 | fails_2
        """
        values = data[col].unique()
        values_array = values.reshape(-1, 1)
        enc = OneHotEncoder(categories='auto', sparse=False)
        values_onehot = enc.fit_transform(values_array)
        return values, values_onehot

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
                
        pfa_onehot = pfa_onehot.groupby(['index']).sum().astype(np.uint16
            ).astype({'outcome': 'bool'}).astype({'outcome': 'uint8'})
        return pfa_onehot

    def _transform_data(self, data, q_matrix, **kwargs):
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
                                             
        # Create onehot representation for skills
        skills, skills_onehot = self._onehot_encoder(df, "skill")
        self.skills = skills
        self.n_skills = len(skills)
        self.params["skills_onehot"] = skills_onehot

        # Create onehot representation for wins, fails and time
        onehot_cols = ["skills_%d" % skill for skill in skills]
        self.cols=["wins", "fails"]
        onehot_cols += self._create_onehot_cols(self.cols)
        self.onehot_cols = onehot_cols
        
        # Parallelize onehot transformation
        n_jobs = kwargs.get("n_jobs", None)
        if n_jobs:
            if n_jobs == -1:
                n_jobs = cpu_count()
            splits = self._get_number_of_splits(df, n_jobs)
            #splits = kwargs.get("splits", 800)
            print("Using %d splits and %d jobs" % (splits, n_jobs))
            # df_split = np.array_split(df, splits)
            #print("Optimizing splits")
            df_split = np.array_split(df, splits)
            with Pool(n_jobs) as pool:
                # a = pool.map(self._apply_onehot, df_split)
                # return a
                pfa_onehot = pd.concat(pool.map(self._apply_onehot, df_split))
        else:
            pfa_onehot = self._apply_onehot(df)
        print("BACK")
        return pfa_onehot

        # n_jobs_sum = kwargs.get("n_jobs_sum", None)
        # if n_jobs_sum:
            # if n_jobs_sum == -1:
                # n_jobs_sum = cpu_count()
            # with Pool(n_jobs_sum) as pool:
                # # Group in one row questions with multiple skills
                # pfa_onehot = pfa_onehot.groupby(['index'])
                # pfa_onehot = pd.concat(pool.map(self._sum_df, pfa_onehot), axis=1).T
                # pfa_onehot = pfa_onehot.drop(columns=['index'])
        # else:
            # # Group in one row questions with multiple skills
            # pfa_onehot = pfa_onehot.groupby(['index']).sum()
            
        pfa_onehot = pfa_onehot.groupby(['index']).sum()
        
        # Change column types to save space and to count just once for outcome result
        #pfa_onehot = pfa_onehot.drop(columns=['skill'] + self.cols)
        pfa_onehot = pfa_onehot.astype(np.uint16).astype({
            'outcome': 'bool'}).astype({'outcome': 'uint8'})        
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
        data = self._transform_data(data, q_matrix, **kwargs)

        # Fit model
        cols = self.onehot_cols
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
        data = self._transform_student_data(data, q_matrix, learning_state)

        # Fit model
        cols = self.onehot_cols
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