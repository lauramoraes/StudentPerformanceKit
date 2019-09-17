import requests
import zipfile
import subprocess
import os
import uuid
import shlex
import numpy as np
import re
import logging
import logging.config

#logging.config.fileConfig(fname='file.conf', disable_existing_loggers=False)

# Get the logger specified in the file
LOGGER = logging.getLogger(__name__)

algorithm_lookup = {
    "bw": "1.1",
    "gd": "1.2",
    "cgd_pr": "1.3.1",
    "cgd_fr": "1.3.2",
    "cgd_hs": "1.3.3"
}


class BKT(object):
    def __init__(self,
                 hmm_folder='hmm-scalable-818d905234a8600a8e3a65bb0f7aa4cf06423f1a',
                 git_commit='818d905234a8600a8e3a65bb0f7aa4cf06423f1a'):

        # Git commit to download hmm-scalable
        self.git_commit = git_commit
        # Set HMM-scalable folder.
        self.hmm_folder = hmm_folder
        # Params
        self.params = None

        # Separate skill params from general model params
        self.model_params = ["SolverId", "nK", "nG", "nS", "nO", "nZ",
                             "Null skill ratios"]

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

    def _create_data_file(self, data, q_matrix):
        if not os.path.exists("hmm_files"):
            os.makedirs("hmm_files")
        filename = "hmm_files/%s" % uuid.uuid4().hex

        # Create data file in the format expected by the tool
        with open("%s.txt" % filename, "w") as step_file:
            for row in data:
                outcome, student_id, question_id = row
                # Transform correct to 1 and incorrect to 2 (requred by the tool)
                outcome = -outcome+2
                skills = np.where(q_matrix[question_id] == 1)
                skills = "~".join(str(skill) for skill in skills[0])
                step_file.write("%s\t%s\t%s\t%s\n" % (outcome, student_id, question_id, skills))
        return filename

    # Probability of being in the learning state given that the student got the
    # question correctly
    def _correct(self, learning_state, skills_idx):
        learning_correct = learning_state * (1 - self.S[skills_idx])
        guess_correct = ((1 - learning_state) * self.G[skills_idx])
        learning_evidence = learning_correct/(learning_correct + guess_correct)
        return learning_evidence

    # Probability of being in the learning state given that the student got the
    # question wrongly
    def _incorrect(self, learning_state, skills_idx):
        learning_incorrect = learning_state * self.S[skills_idx]
        guess_incorrect = ((1 - learning_state) * (1 - self.G[skills_idx]))
        learning_evidence = learning_incorrect/(learning_incorrect + guess_incorrect)
        return learning_evidence

    # Update learning state probability
    def _update(self, learning_state, skills_idx, iscorrect=True):
        if iscorrect:
            learning_evidence = self._correct(learning_state, skills_idx)
        else:
            learning_evidence = self._incorrect(learning_state, skills_idx)
        learning_state = learning_evidence + ((1 - learning_evidence) * self.T[skills_idx])
        return learning_state

    # Predict whether the next question will be answered correctly or
    # incorrectly by the student
    def _get_correct_prob(self, learning_state, skills_idx):
        return (1-self.S[skills_idx])*learning_state + self.G[skills_idx]*(1-learning_state)

    def download(self):
        """  This implementation is a wrapper around the
        HMM-scalable tool ( http://yudelson.info/hmm-scalable).
        This function will download and install the original implementation.

        Returns
        -------
        self: object

        Notes
        -----
        This is a wrapper around the HMM-scalable tool
        (http://yudelson.info/hmm-scalable).  """

        # Download zipfile from GitHub
#         results = requests.get('https://github.com/myudelson/hmm-scalable/archive/master.zip')
        results = requests.get('https://github.com/myudelson/hmm-scalable/archive/%s.zip' % self.git_commit)
        with open('/tmp/hmm-scalable.zip', 'wb') as f:
            f.write(results.content)

        # Extract zipfile
        file = zipfile.ZipFile('/tmp/hmm-scalable.zip')
        file.extractall(path='.')

        # Install
        process = subprocess.Popen("make all", stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=self.hmm_folder, shell=True)
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            raise RuntimeError("Could not build HMM tool. Check if the make utility is installed "
                               "and if the folder has appropriate permissions.\n "
                               "Code: %d\n"
                               "Error: %s" % (process.returncode, stderr))

    def fit(self, data, q_matrix, solver='bw', iterations=200):
        """ Fit BKT model to data.
        As of July 2019, just default parameters are allowed.

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

        solver: string, optional
            Algorithm used to fit the BKT model. Available solvers are:
            'bw': Baum-Welch (default)
            'gd': Gradient Descent
            'cgd_pr': Conjugate Gradient Descent (Polak-Ribiere)
            'cgd_fr': Conjugate Gradient Descent (Fletcher–Reeves)
            'cgd_hs': Conjugate Gradient Descent (Hestenes-Stiefel)

        iterations: integer, optional
            Maximum number of iterations

        Returns
        -------
        self: object

        Notes
        -----
        This is a wrapper around the HMM-scalable tool (http://yudelson.info/hmm-scalable)
        """
        # Create data file
        filename = self._create_data_file(data, q_matrix)

        # Run train program
        command = "./trainhmm -s %s -i %d -d ~ ../%s.txt ../%s_model.txt" % (
            algorithm_lookup[solver], iterations, filename, filename)
        args = shlex.split(command)
        process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=self.hmm_folder)
        process.wait()
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            raise RuntimeError("Could not train HMM model. Check if the HMM files are properly created and "
                               "accessible.\n"
                               "Code: %d\n"
                               "Error: %s" % (process.returncode, stderr))

        # Extract fitted params
        with open("%s_model.txt" % filename, "r") as model_file:
            content = model_file.read()
        params = {"skills": []}
        params_regex = re.findall(r'^([\w ]+)\t(.*)$', content, flags=re.M)
        idx = 0
        while idx < len(params_regex):
            param, value = params_regex[idx]
            # If param is not listed in model params, it's a skill param
            if param not in self.model_params:
                params["skills"].append({
                    "skill": value,
                    # Get PI matrix
                    "priors": np.asarray([float(i) for i in params_regex[idx+1][1].split("\t")]),
                    # Get A matrix
                    "transitions": np.asarray([float(i) for i in params_regex[idx+2][1].split("\t")]),
                    # Get B matrix
                    "emissions": np.asarray([float(i) for i in params_regex[idx+3][1].split("\t")])
                })
                LOGGER.debug("Appending skill %s params: %s" % (value, params["skills"]))
                idx += 4
            else:
                params[param] = value
                idx += 1

        self.params = params
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

        learning_state: {array-like}, shape (n_skills,)
            Set learning state for each skill. The skills should be in the same
            order as in the model parameters.

        Returns
        -------
        outcome : {array-like}, shape (n_steps, 2)
            Outcome probabilites for steps in data. Column 0 corresponds to
            outcome 0 (incorrect) and column 1 to outcome 1 (correct)

        Notes
        -----
        This calculates the predicted steps in the same way it is done in the
        HMM-scalable tool (http://yudelson.info/hmm-scalable)

        """

        # Get model params
        model_params = self.get_params()
        self.n_skills = len(model_params['skills'])

        # Construct params matrices
        skills = np.zeros(self.n_skills)

        # If learning state is not set, use prior probabilities
        if not learning_state:
            learning_state = np.zeros(self.n_skills)

        self.T = np.zeros(self.n_skills)
        self.S = np.zeros(self.n_skills)
        self.G = np.zeros(self.n_skills)
        for idx, skill in enumerate(model_params['skills']):
            skills[idx] = skill['skill']

            # Update learning using priors if it was not set
            if learning_state is None:
                learning_state[idx] = skill['priors'][0]

            # Update params
            self.T[idx] = skill['transitions'][2]
            self.S[idx] = skill['emissions'][1]
            self.G[idx] = skill['emissions'][2]

        data = np.asarray(data)
        self.outcomes = data[:, 0]
        self.n_questions = len(self.outcomes)
        self.loglikelihood = np.zeros(self.n_questions)
        outcome_prob_skill = np.zeros((self.n_skills, self.n_questions))
        self.outcome_prob = np.zeros((self.n_questions, 2))

        for idx, outcome in enumerate(self.outcomes):
            # Get question skills from q_matrix
            question_id = data[idx, 1]

            # Get skill index in skills list
            skills_name = np.where(q_matrix[question_id, :] == 1)
            skills_idx = np.where(np.isin(skills, skills_name))

            # Sliced learning states (skill is present in this question)
            l_sliced = learning_state[skills_idx]

            # Calculate chance of being correct for each skill
            outcome_prob_skill[skills_idx, idx] = self._get_correct_prob(l_sliced, skills_idx)
            # Column 0 is probability of failing and column 1 is probability of
            # success
            self.outcome_prob[idx, 1] = outcome_prob_skill[:, idx].sum()/skills_idx[0].shape[0]
            self.outcome_prob[idx, 0] = 1-self.outcome_prob[idx, 1]

            # Calculate LL and update state
            if outcome == 1:
                ll_local = np.log(self.outcome_prob[idx, 1])
                l_sliced = self._update(l_sliced, skills_idx, True)
            else:
                ll_local = np.log(self.outcome_prob[idx, 0])
                l_sliced = self._update(l_sliced, skills_idx, False)
            learning_state[skills_idx] = l_sliced
            self.loglikelihood[idx] += ll_local

        return self.outcome_prob

    def predict_proba(self, data, q_matrix, learning_state=None):
        """ Predict student outcome probabilities based on trained model.

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

        learning_state: {array-like}, shape (n_skills,)
            Set learning state for each skill. The skills should be in the same
            order as in the model parameters.

        Returns
        -------
        outcome : {array-like}, shape (n_steps, 2)
            Outcome probabilites for steps in data. Column 0 corresponds to
            outcome 0 (incorrect) and column 1 to outcome 1 (correct)

        Notes
        -----
        This calculates the predicted steps in the same way it is done in the
        HMM-scalable tool (http://yudelson.info/hmm-scalable)
        """
        y_pred_proba = self._predict(data, q_matrix, learning_state)
        return y_pred_proba

    def predict(self, data, q_matrix, model_file=None):
        """ Predict student outcomes based on trained model. This is just the
        hard-assigment (highest probability) of the outcome probabilities.

        Parameters
        ----------
        data : {array-like}, shape (n_steps, 3)
            Sequence of students steps. Each of the three dimensions are:
            Observed outcome: 0 for fail and 1 for success
            Student id: student unique identifier
            Question id: question id in q_matrix

        Returns
        -------
        outcome : {array-like}, shape (n_steps,)
            Outcomes for steps in data. Outcome 0 is incorrect and outcome 1 is
            correct.

        Notes
        -----
        This calculates the predicted steps in the same way it is done in the
        HMM-scalable tool (http://yudelson.info/hmm-scalable)
        """
        y_pred_proba = self._predict(data, q_matrix, model_file)
        y_pred = np.argmax(y_pred_proba, axis=1)
        return y_pred

    def score(self):
        """
        Calculates AIC, BIC, RMSE and accuracy for the predicted sample

        """
        # AIC
        self.aic = -2*self.loglikelihood.sum() + 2*4*self.n_skills

        # BIC
        self.bic = -2*self.loglikelihood.sum() + 4*self.n_skills*np.log(
            self.n_questions)

        # RMSE
        rmse = ((1-self.outcome_prob[range(self.n_questions), self.outcomes])**2).sum()
        self.rmse = np.sqrt(rmse/self.n_questions)

        # Accuracy
        estimated_outcome = np.argmax(self.outcome_prob, axis=1)
        self.acc = (estimated_outcome == self.outcomes).sum()/self.n_questions

        return self.aic, self.bic, self.rmse, self.acc

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
