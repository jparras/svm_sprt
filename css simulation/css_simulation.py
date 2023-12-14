import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from matplotlib2tikz import save as tikz_save
import pickle
from tabulate import tabulate


def features_extraction(x):  # Used to extract the features for the 1 class SVM
    x -= np.mean(x)
    x = np.abs(np.fft.fft(np.correlate(x, x, 'full')))
    return x / np.amax(x)


def train_svm(theta, A, B, nu=0.1, tryouts=5, training_size=500, val_size=500, window_size=5, num_cores=4, verbose=True):
    # Get the SVM with lower error using parallel processing
    if verbose:
        print('Training OC-SVM...')
    def train_svm():
        from sklearn import svm
        svm = svm.OneClassSVM(nu=nu, gamma='auto')  # Create svm
        sample_size = 2 * window_size - 1  # Size of the autocorrelation vector
        # Train the SVM
        x = np.random.binomial(1, theta, size=training_size)
        t = np.arange(len(x))
        x = np.cumsum(x) * A + (t + 1) * B
        number_of_seqs = len(x) - window_size
        batch_train = np.zeros([number_of_seqs, sample_size])
        for i in range(number_of_seqs):
            batch_train[i, :] = features_extraction(x[i: i + window_size])
        svm.fit(batch_train)
        y_pred_train = svm.predict(batch_train)

        # Validation
        x_val = np.random.binomial(1, theta, size=val_size)
        t = np.arange(len(x_val))
        x_val = np.cumsum(x_val) * A + (t + 1) * B
        number_of_seqs = len(x_val) - window_size
        batch_val = np.zeros([number_of_seqs, sample_size])
        for i in range(number_of_seqs):
            batch_val[i, :] = features_extraction(x_val[i: i + window_size])
        y_pred_val = svm.predict(batch_val)

        training_error = y_pred_train[y_pred_train == -1].size / len(x)
        pred_error = y_pred_val[y_pred_val == -1].size / len(x_val)
        return [svm, training_error, pred_error]

    #out = Parallel(n_jobs=num_cores, verbose=5)(delayed(train_svm)() for _ in range(tryouts))
    out = []
    for _ in range(tryouts):
        out.append(train_svm())

    total_error_vector = np.array([out[i][1] + out[i][2] for i in range(tryouts)])
    if verbose:
        print('SVM trained results: ', '\n',
              'Nu: ', nu, '\n',
              'Total error values = ', total_error_vector, '\n',
              'Training error values = ', np.array([out[i][1] for i in range(tryouts)]), '\n',
              'Prediction error values = ', np.array([out[i][2] for i in range(tryouts)]), '\n',
              'Minimum total error value = ', np.amin(total_error_vector))
    return out[np.argmin(total_error_vector)][0]  # Return the SVM with minimal total error


class attacker():
    def __init__(self, params):
        self.attack_type = params['attack_type']
        self.n_of_attackers = params['n_of_attackers']
        self.n_of_normal_sensors = params['n_of_sensors'] - params['n_of_attackers']
        assert self.n_of_normal_sensors >= 0  # Check that there are enough attackers!
        self.n_of_sensors = params['n_of_sensors']
        self.mask = np.zeros(self.n_of_sensors).astype(bool)
        self.mask[np.random.choice(self.n_of_sensors, size=self.n_of_attackers, replace=False)] = 1
        if self.attack_type == 'intelligent':
            self.n = np.zeros(self.n_of_sensors)
            self.s = np.zeros(self.n_of_sensors)
            self.att_h = np.log((1 - params['beta']) / params['alpha'])
            theta_0 = (1 - params['pe']) * params['duty_cycle'] + params['pe'] * (1 - params['duty_cycle'])
            theta_1 = theta_0 + params['svm_sensitivity']
            self.att_A = np.log((theta_1 * (1 - theta_0)) / (theta_0 * (1 - theta_1)))
            self.att_B = np.log((1 - theta_1) / (1 - theta_0))


    def obtain_attack(self, sensed_value, sensor_index): # Method that performs the byzantine attack
        if self.mask[sensor_index]:  # If true, it is an attacker
            if self.attack_type == 'always_no':
                sensed_value = 0
            elif self.attack_type == 'always_yes':
                sensed_value = 1
            elif self.attack_type == 'always_false':
                sensed_value = 1 - sensed_value
            elif self.attack_type == 'intelligent':
                # g1: LLR value seen by the defense mechanism if 1 is used!
                g1 = (self.s[sensor_index] + 1) * self.att_A + (self.n[sensor_index] + 2) * self.att_B
                if g1 < self.att_h:
                    sensed_value = 1
                else:
                    sensed_value = 0
                self.n[sensor_index] += 1
                self.s[sensor_index] += sensed_value
            elif self.attack_type == 'none':
                pass
            else:
                raise RuntimeError('Attack not recognizec')
        return sensed_value

    def get_attackers(self):
        return self.mask

class defender():
    def __init__(self, params):
        self.defense_type = params['defense_type']
        self.fusion_type = params['fusion_type']
        self.n_of_sensors = params['n_of_sensors']
        self.banned_sensors = np.zeros(self.n_of_sensors).astype(bool)

        # SVM defense mechanism parameters
        self.svm = params['svm']
        self.svm_window_size = params['svm_window_size']
        self.use_svm = True  # Flag: whether to use toe OCSVM additinal defense mechanism
        self.svm_h = np.log((1 - params['beta']) / params['alpha'])
        self.svm_l = np.log(params['beta'] / (1 - params['alpha']))
        theta_0 = (1 - params['pe']) * params['duty_cycle'] + params['pe'] * (1 - params['duty_cycle'])
        theta_1 = theta_0 + params['svm_sensitivity']
        self.svm_A = np.log((theta_1 * (1 - theta_0)) / (theta_0 * (1 - theta_1)))
        self.svm_B = np.log((1 - theta_1) / (1 - theta_0))
        self.LLR_n = [[self.svm_B] for _ in range(self.n_of_sensors)]  # Initialize the LLR vector
        self.svm_delta = params['svm_delta']

        # Majority rule parameters
        self.maj_n = params['maj_n']  # Maximum sample size for voting rule
        self.maj_k = int(self.maj_n / 2.0 + 1)  # Decission threshold
        self.maj_reports = [0, 0]

        # SPRT voting rule parameters
        self.sprt_h = np.log((1 - params['beta']) / params['alpha'])
        self.sprt_l = np.log((params['beta']) / (1 - params['alpha']))
        self.sprt_A1 = np.log((1 - params['pe']) / params['pe'])
        self.sprt_A0 = np.log(params['pe'] / (1 - params['pe']))
        self.sprt_test_statistic = 0

        # WSPRT voting rule parameters
        self.wsprt_reputations = np.zeros(self.n_of_sensors)
        self.wsprt_weights = self.obtain_wsprt_weight()
        self.wsprt_h = np.log((1 - params['beta']) / params['alpha'])
        self.wsprt_l = np.log((params['beta']) / (1 - params['alpha']))
        self.wsprt_A1 = np.log((1 - params['pe']) / params['pe'])
        self.wsprt_A0 = np.log(params['pe'] / (1 - params['pe']))
        self.wsprt_test_statistic = 0
        self.wsprt_historic = []  # VEctor used to update reputations

        # EWSPRT voting rule parameters
        self.ewsprt_reputations = np.zeros(self.n_of_sensors)
        self.ewsprt_weights = self.obtain_ewsprt_weight()
        self.ewsprt_h = np.log((1 - params['beta']) / params['alpha'])
        self.ewsprt_l = np.log((params['beta']) / (1 - params['alpha']))
        self.ewsprt_A1 = np.log((1 - params['pe']) / params['pe'])
        self.ewsprt_A0 = np.log(params['pe'] / (1 - params['pe']))
        self.ewsprt_test_statistic = 0
        self.ewsprt_historic = []  # VEctor used to update reputations

        self.n_max_steps = params['n_max_steps']  # Maximum length of simulation
        self.reports = [[] for _ in range(self.n_of_sensors)]  # Initialize the storic of reports
        self.n = 0

    def obtain_wsprt_weight(self):
        weights = np.zeros(self.n_of_sensors)
        for i in range(self.n_of_sensors):
            if self.wsprt_reputations[i] > -5:
                weights[i] = (self.wsprt_reputations[i] + 5) / (np.amax(self.wsprt_reputations) + 5)
        return weights

    def obtain_ewsprt_weight(self):
        weights = np.zeros(self.n_of_sensors)
        for i in range(self.n_of_sensors):
            if self.wsprt_reputations[i] > -5:
                weights[i] = (self.wsprt_reputations[i] + 5) / (np.mean(self.wsprt_reputations) + 5)
        return weights

    def defense(self, sensor_index):  # Apply the defense mechanism to a report
        if not self.banned_sensors[sensor_index]:  # Proceed only if sensor is not banned
            if self.defense_type == 'none':
                pass
            elif self.defense_type == 'sprt' or self.defense_type == 'svm_sprt':
                report = self.reports[sensor_index][:]
                if self.defense_type == 'svm_sprt':
                    svm_output = 0 # Initialize the svm element
                    if len(report) >= self.svm_window_size:
                        # Invoque the SVM mechanism
                        tv = np.array(report[-self.svm_window_size:])
                        t = np.arange(self.n - self.svm_window_size, self.n)
                        tv = np.cumsum(tv) * self.svm_A + (t + 1) * self.svm_B
                        correlator = self.svm.predict(features_extraction(tv).reshape([1, 2 * self.svm_window_size - 1]))
                        if correlator == -1:
                            svm_output = 1
                    svm_llr_value = svm_output * self.svm_delta * (self.svm_A + self.svm_B)
                else:
                    svm_llr_value = 0  # No SVM is used
                # SPRT test
                if report[-1] == 0:
                    self.LLR_n[sensor_index].append(self.LLR_n[sensor_index][-1] + self.svm_B + svm_llr_value)
                else:
                    self.LLR_n[sensor_index].append(self.LLR_n[sensor_index][-1] + self.svm_A + self.svm_B + svm_llr_value)

                if self.LLR_n[sensor_index][-1] >= self.svm_h:  # Ban sensor if threshold is surpasses
                    self.banned_sensors[sensor_index] = True
            else:
                raise RuntimeError('Defense type unknown')
        else:
            raise RuntimeError('SVM mechanism called with a banned sensor!')

    def fusion_step(self, report, sensor_index):
        decision_taken = -1  # -1: No decision, 0: no primary present, 1: primary present
        if self.fusion_type == 'maj':  # Majority rule
            self.maj_reports[report] += 1
            if self.maj_reports[0] >= self.maj_k:
                decision_taken = 0
            elif self.maj_reports[1] >= self.maj_k:
                decision_taken = 1
        elif self.fusion_type == 'sprt':
            # Update the test statistic (log-probability!)
            if report == 0:
                self.sprt_test_statistic += self.sprt_A0
            elif report == 1:
                self.sprt_test_statistic += self.sprt_A1
            else:
                raise NotImplementedError

            if self.sprt_test_statistic >= self.sprt_h:
                decision_taken = 1
            elif self.sprt_test_statistic <= self.sprt_l:
                decision_taken = 0

        elif self.fusion_type == 'wsprt':
            self.wsprt_historic.append([sensor_index, report])
            # Update the test statistic (log-probability!)
            if report == 0:
                self.wsprt_test_statistic += self.wsprt_A0
            elif report == 1:
                self.wsprt_test_statistic += self.wsprt_A1
            else:
                raise NotImplementedError

            if self.wsprt_test_statistic >= self.wsprt_h:
                decision_taken = 1
            elif self.wsprt_test_statistic <= self.wsprt_l:
                decision_taken = 0
            if decision_taken >= 0:
                # Update reputations
                for info in self.wsprt_historic:
                    if info[1] == decision_taken:
                        self.wsprt_reputations[info[0]] += 1
                    else:
                        self.wsprt_reputations[info[0]] -= 1
        elif self.fusion_type == 'ewsprt':
            self.ewsprt_historic.append([sensor_index, report])
            # Update the test statistic (log-probability!)
            if report == 0:
                self.ewsprt_test_statistic += self.ewsprt_A0
            elif report == 1:
                self.ewsprt_test_statistic += self.ewsprt_A1
            else:
                raise NotImplementedError

            if self.ewsprt_test_statistic >= self.ewsprt_h:
                decision_taken = 1
            elif self.ewsprt_test_statistic <= self.ewsprt_l:
                decision_taken = 0
            if decision_taken >= 0:
                # Update reputations
                for info in self.ewsprt_historic:
                    if info[1] == decision_taken:
                        self.ewsprt_reputations[info[0]] += 1
                    else:
                        self.ewsprt_reputations[info[0]] -= 1
        else:
            raise RuntimeError('Fusion type unkwnown')
        return decision_taken

    def fusion_reset(self):  # Reset all fusion parameters
        # Majority rule reset
        self.maj_reports = [0, 0]
        # SPRT reset
        self.sprt_test_statistic = 0
        # WSPRT reset
        self.wsprt_weights = self.obtain_wsprt_weight()
        self.wsprt_historic = []
        self.wsprt_test_statistic = 0
        # EWSPRT reset
        self.ewsprt_weights = self.obtain_ewsprt_weight()
        self.ewsprt_historic = []
        self.ewsprt_test_statistic = 0

    def get_banned_sensors(self):
        return self.banned_sensors

    def add_report(self, report, sensor_index):
        self.reports[sensor_index].append(report)


class simulate_css():
    def __init__(self, params):
        self.params = params
        self.attacker = None
        self.defender = None
        self.params['svm'] = self.obtain_svm()
        self.simulation_results = [[] for _ in range(self.params['n_simulations'])]

    def obtain_svm(self):
        svm = None
        if self.params['defense_type'] == 'svm_sprt':
            theta_0 = (1 - self.params['pe']) * self.params['duty_cycle'] + self.params['pe'] * (1 - self.params['duty_cycle'])
            theta_1 = theta_0 + self.params['svm_sensitivity']
            svm_A = np.log((theta_1 * (1 - theta_0)) / (theta_0 * (1 - theta_1)))
            svm_B = np.log((1 - theta_1) / (1 - theta_0))
            svm = train_svm(theta_0, svm_A, svm_B, nu=self.params['svm_nu'],
                            tryouts=self.params['svm_tryouts'], training_size=self.params['svm_training_size'],
                            val_size=self.params['svm_validation_size'], window_size=self.params['svm_window_size'],
                            num_cores=self.params['num_cores'], verbose=self.params['verbose'])
        return svm


    def obtain_attacker(self):
        self.attacker = attacker(self.params)

    def obtain_defender(self):
        self.defender = defender(self.params)

    def select_sensor(self, banned_sensors, step):
        if self.params['fusion_type'] == 'ewsprt':
            available_sensors = []
            reputations = []
            for i in range(self.params['n_of_sensors']):
                if not banned_sensors[i]:
                    available_sensors.append(i)
                    reputations.append(self.defender.ewsprt_reputations[i])
            sorted_indexes = sorted(range(len(reputations)), key=lambda k: reputations[k])
            return available_sensors[sorted_indexes[(step - 1) % len(sorted_indexes)]]
        else:  # Return a random sensor
            return np.random.choice(np.where(banned_sensors == False)[0])  # Return a random sensor not banned

    def simulate(self):
        for sim in range(self.params['n_simulations']):
            self.obtain_attacker()
            self.obtain_defender()
            for step in range(self.params['n_max_steps']):
                if self.params['verbose'] and step % 50 == 0:
                    print(' Simulation ', sim + 1, ' of ', self.params['n_simulations'], '; step ', step + 1, ' of ',
                          self.params['n_max_steps'])
                # Reset fusion schemes
                self.defender.fusion_reset()
                channel_state = np.random.binomial(1, self.params['duty_cycle'])
                decision_taken = False
                n_of_steps_to_decide = 0
                while not decision_taken:
                    n_of_steps_to_decide += 1
                    banned_sensors = self.defender.get_banned_sensors()  # Update list of banned sensors
                    sensor_index = self.select_sensor(banned_sensors, n_of_steps_to_decide)  # Sensor to ask for a report
                    error_commited = np.random.binomial(1, self.params['pe'])  # The sensor makes an error!
                    if error_commited:
                        sensed_value = 1 - channel_state
                    else:
                        sensed_value = channel_state
                    report = self.attacker.obtain_attack(sensed_value, sensor_index)  # Obtain report
                    self.defender.add_report(report, sensor_index)  # Send report to the defense mechanism
                    # Update defense mechanism
                    self.defender.defense(sensor_index)
                    banned_sensors = self.defender.get_banned_sensors()  # Update list of banned sensors
                    if not banned_sensors[sensor_index]:
                        # Decision fusion
                        decision = self.defender.fusion_step(report, sensor_index)
                        if decision >= 0:
                            decision_taken = True
                        banned_sensors = self.defender.get_banned_sensors()  # Update list of banned sensors
                        if np.sum(banned_sensors) == self.params['n_of_sensors']:
                            decision_taken = True
                            decision = 1  # Do not transmit: conservative decision
                    else:
                        pass # Do not use the report of this sensor
                    if n_of_steps_to_decide >= self.params['max_fusion_steps'] or \
                            np.sum(banned_sensors) == self.params['n_of_sensors']:
                        decision_taken = True
                        decision = 1 # Do not transmit: conservative decision
                results = {'ch_state': channel_state,
                           'decision': decision,
                           'decision_error': np.abs(channel_state - decision) > 0.5,
                           'n_of_steps_to_decide': n_of_steps_to_decide,
                           'banned_sensors': self.defender.get_banned_sensors(),
                           'actual_attackers': self.attacker.get_attackers()}
                self.simulation_results[sim].append(results)
                if np.sum(banned_sensors) == self.params['n_of_sensors']:
                    break  # Break inner loop, for all sensors have been banned from the network

    def obtain_results(self):
        error = []
        steps_to_decide = []
        attackers_banned = []
        attackers_not_banned = []
        good_sensors_banned = []
        good_sensors_not_banned = []
        for sim in range(self.params['n_simulations']):
            for result in self.simulation_results[sim]:
                error.append(result['decision_error'])
                steps_to_decide.append(result['n_of_steps_to_decide'])
            # Banned sensors measures correspond to the final values
            banned_sensors = self.simulation_results[sim][-1]['banned_sensors']
            actual_attackers = self.simulation_results[sim][-1]['actual_attackers']
            attackers_banned.append(np.sum(np.logical_and(banned_sensors, actual_attackers)))
            attackers_not_banned.append(np.sum(np.logical_and(np.logical_not(banned_sensors), actual_attackers)))
            good_sensors_banned.append(np.sum(np.logical_and(banned_sensors, np.logical_not(actual_attackers))))
            good_sensors_not_banned.append(np.sum(np.logical_and(np.logical_not(banned_sensors), np.logical_not(actual_attackers))))

        if self.params['verbose']:
            print('SIMULATION RESULTS')
            print('Number of simulations: ', self.params['n_simulations'])
            print('Number of steps per simulation', self.params['n_max_steps'])
            print('Attack mechanism: ', self.params['attack_type'])
            print('Defense mechanism: ', self.params['defense_type'])
            print('Fusion mechanism: ', self.params['fusion_type'])
            print('Duty cycle: ', self.params['duty_cycle'], '; pe: ', self.params['pe'], '; theta: ', self.params['theta'])
            print('Mean decision error: ', np.mean(error))
            print('Mean number of sensors called to decide: ', np.mean(steps_to_decide))
            print('Mean number of attackers banned: ', np.mean(attackers_banned))
            print('Mean number of attackers not banned: ', np.mean(attackers_not_banned))
            print('Mean number of good sensors banned: ', np.mean(good_sensors_banned))
            print('Mean number of good sensors not banned: ', np.mean(good_sensors_not_banned))

            # Plot example of SPRT advance in SVM
            if self.params['fusion_type'] == 'sprt' or self.params['fusion_type'] == 'sprt':
                attackers = self.attacker.get_attackers()
                for i in range(self.params['n_of_sensors']):
                    if attackers[i]:
                        plt.plot(self.defender.LLR_n[i], 'r')
                    else:
                        plt.plot(self.defender.LLR_n[i], 'b')
                plt.show()

        results_dictionary = {'n_simulations': self.params['n_simulations'],
                              'n_max_steps': self.params['n_max_steps'],
                              'attack_type': self.params['attack_type'],
                              'defense_type': self.params['defense_type'],
                              'fusion_type': self.params['fusion_type'],
                              'duty_cycle': self.params['duty_cycle'],
                              'pe': self.params['pe'],
                              'n_of_sensors': self.params['n_of_sensors'],
                              'n_of_attackers': self.params['n_of_attackers'],
                              'mean_decision_error': np.mean(error),
                              'mean_steps_to_decide': np.mean(steps_to_decide),
                              'mean_attackers_banned': np.mean(attackers_banned),
                              'mean_attackers_not_banned': np.mean(attackers_not_banned),
                              'mean_good_sensors_banned': np.mean(good_sensors_banned),
                              'mean_good_sensors_not_banned': np.mean(good_sensors_not_banned),
                              'params': self.params}
        return results_dictionary


if __name__ == '__main__':
    # Simulation flag: whther to obtain the CSS data
    simulate = True
    # Parameters
    params={'attack_type': 'intelligent',
            'defense_type': 'svm_sprt',
            'fusion_type': 'maj',
            'n_of_attackers': 5,
            'n_of_sensors': 10,
            'num_cores': 7, # Multiprocessing
            'verbose': False,
            'n_max_steps': 50,  # Max number of steps to simulate
            'n_simulations': 50,
            'duty_cycle': 0.5,  # Transmission probability for the primary
            'pe': 0.3,  # Error probability for sensors measurements
            'max_fusion_steps': 20, # Max steps before trunction for fusion rules
            'alpha': 0.05, # When applicable, type I error probability
            'beta': 0.05, # When applicable, type II error probability
            }
    # SVM parameters
    params['svm_nu'] = 0.1  # SVM parameter
    params['svm_tryouts'] = 20  # Number of SVMs to train
    params['svm_training_size'] = 500
    params['svm_validation_size'] = 500
    params['svm_window_size'] = 5
    params['svm_delta'] = 0.05  # SVM parameter
    params['svm_sensitivity'] = 0.1  # SVM SPRT discriminator sensitivity
    # Majority rules parameters
    params['maj_n'] = params['max_fusion_steps']  # Sample size: same as max number of steps

    # Generate combinations of parameters for simulation
    keys = ['attack_type', 'defense_type', 'fusion_type', 'n_of_attackers', 'pe']
    vals = [['always_yes', 'always_no', 'always_false', 'intelligent'],
            ['none', 'sprt', 'svm_sprt'],
            ['maj', 'ewsprt'],
            [0, 1, 2, 3, 4, 5],
            [0.1, 0.2, 0.3]]
    values = [(x1, x2, x3, x4, x5) for x1 in vals[0] for x2 in vals[1] for x3 in vals[2] for x4 in vals[3] for x5 in vals[4]]

    params_vector = []
    for value in values:
        params_copy = params.copy()  # This copy is needed, otherwise there is only one dictionary!
        for key in keys:
            params_copy[key] = value[keys.index(key)]
        params_vector.append(params_copy)

    def process_simulation(params):
        simulator = simulate_css(params)
        simulator.simulate()
        return simulator.obtain_results()

    if simulate:
        sim_results = Parallel(n_jobs=params['num_cores'], verbose=5)(delayed(process_simulation)(par)
                                                                      for par in params_vector)

        # Save
        with open('results_css.pickle', 'wb') as handle:
            pickle.dump(sim_results, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Show total results
    with open('results_css.pickle', 'rb') as handle:
        sim_results = pickle.load(handle)
    res = []
    headers = ['attack_type', 'defense_type', 'fusion_type', 'duty_cycle', 'pe', 'n_of_sensors', 'n_of_attackers',
               'mean_decision_error', 'mean_steps_to_decide', 'mean_attackers_banned', 'mean_attackers_not_banned',
               'mean_good_sensors_banned', 'mean_good_sensors_not_banned']
    for result in sim_results:
        res.append([result[key] for key in headers])

    print(tabulate(res, headers=headers, tablefmt='orgtbl'))

    # Plot data
    for pe in vals[4]:
        for attack in vals[0]:  # For each attack
            mk = ['o', 'v', 's']
            fusion_color = ['r', 'g', 'b', 'k']
            mde = -np.ones([len(vals[1]), len(vals[2]), len(vals[3])])  # Defense x fusion schemes x attackers
            alt = -np.ones([len(vals[1]), len(vals[2]), len(vals[3])])  # Defense x fusion schemes x attackers
            for r in sim_results:
                if r['attack_type'] == attack and r['pe'] == pe:
                    mde[vals[1].index(r['defense_type']), vals[2].index(r['fusion_type']),
                        vals[3].index(r['n_of_attackers'])] = r['mean_decision_error']
                    alt[vals[1].index(r['defense_type']), vals[2].index(r['fusion_type']),
                        vals[3].index(r['n_of_attackers'])] = r['mean_steps_to_decide']
            for def_idx in range(len(vals[1])):
                for fus_idx in range(len(vals[2])):
                    plt.plot(vals[3], mde[def_idx, fus_idx, :], marker=mk[def_idx], fillstyle='none',
                             color=fusion_color[fus_idx], label=vals[1][def_idx] + ' ' + vals[2][fus_idx])
            plt.title('Attack mode: ' + attack + '; pe = ' + str(pe))
            plt.legend(loc='best')
            plt.xlabel('Number of attackers')
            plt.ylabel('Decision error')
            tikz_save('css_error_' + attack + str(pe) + '.tikz', figureheight='\\figureheight',
                      figurewidth='\\figurewidth')
            plt.show()

            for def_idx in range(len(vals[1])):
                for fus_idx in range(len(vals[2])):
                    plt.plot(vals[3], alt[def_idx, fus_idx, :], marker=mk[def_idx], fillstyle='none',
                             color=fusion_color[fus_idx], label=vals[1][def_idx] + ' ' + vals[2][fus_idx])
            plt.title('Attack mode: ' + attack + '; pe = ' + str(pe))
            plt.legend(loc='best')
            plt.xlabel('Number of attackers')
            plt.ylabel('Average test lenght')
            tikz_save('css_arl_' + attack + str(pe) + '.tikz', figureheight='\\figureheight',
                      figurewidth='\\figurewidth')
            plt.show()

