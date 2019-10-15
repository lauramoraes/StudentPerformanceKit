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
    "sklearn": {"fit_intercept": False},
    "sm": {"family": sm.families.Binomial()}
}

sm_regularized_params = ["alpha", "start_params", "refit"]


class PFA(object):
    def __init__(self):
        # Model
        self.model = None
        # Params
        self.params = None

    @staticmethod
    def _create_onehot(row, skills, skills_onehot):
        """ Transform each row to its one hot version """
        idx = np.where(skills == row['skill'])[0]
        wins = row['wins']*skills_onehot[idx][0]
        fails = row['fails']*skills_onehot[idx][0]
        return np.concatenate((skills_onehot[idx][0], wins, fails))

    def _skills_onehot(self, data):
        """ Transform PFA data to its onehot version where each columns
        represents a skill, wins and fails for the corresponding skills. A
        table header for 2 skills would contain the following information
        skill_1 | skill_2 | wins_1 | wins_2 | fails_1 | fails_2
        """
        skills = data['skill'].unique()
        skills_array = skills.reshape(-1, 1)
        enc = OneHotEncoder(categories='auto', sparse=False)
        skills_onehot = enc.fit_transform(skills_array)
        onehot_array = data.apply(self._create_onehot, axis=1,
                                  args=(skills, skills_onehot))
        cols = ["skills_%d"%skill for skill in skills]
        cols += ["wins_%d"%skill for skill in skills]
        cols += ["fails_%d"%skill for skill in skills]
        onehot_df = pd.DataFrame(onehot_array.tolist(), columns=cols)
        data = pd.concat((data, onehot_df), axis=1)
        return data, cols

    def _transform_data(self, data, q_matrix):
        """ Transform original data into PFA expected format. Calculates wins,
        fails, get skills and transform everything into one-hot variables """
        skills_count = {}
        pfa_data = []
        for row in data:
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
                pfa_row = (skill, wins, fails, outcome)
                pfa_data.append(pfa_row)

                # Update wins or fails counter
                if outcome == 1:
                    skills_count[student_id][skill]["wins"] += 1
                else:
                    skills_count[student_id][skill]["fails"] += 1

        # Create dataframe from PFA data
        df = pd.DataFrame(pfa_data, columns=["skill", "wins", "fails",
                                             "outcome"])

        pfa_onehot = self._skills_onehot(df)
        return pfa_onehot

    def _create_data(data):
        idx = np.where(skills == row['skill'])[0][0]
        wins = row['wins']*skills_onehot[idx]
        fails = row['fails']*skills_onehot[idx]
        return np.concatenate((skills_onehot[idx], wins, fails))

    def fit(self, data, q_matrix, lib='sklearn', **kwargs):
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

        lib: sklearn uses Scikit-Learn LogisticRegression module. sm uses
            StatsModel glm module. sm only supports elasticnet penalty.

        kwargs = extra parameters to be passed to sklearn or sm libraries. If
            penalty parameter is set to sm library, fit_regularized method is
            used.

        Returns
        -------
        self: object

        """
        # Transform data to PFA format
        data, cols = self._transform_data(data, q_matrix)

        # Fit model
        X = data[cols]
        X = sm.add_constant(X, prepend=True)
        y = data['outcome']

        # Fit data
        params = copy.deepcopy(default_params[lib])
        params.update(kwargs)

        if lib == "sklearn":
            model = algorithm_lookup[lib](**params)
            model.fit(X, y)
        elif lib == "sm":
            model = algorithm_lookup[lib](y, X, **params)
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

        return model
