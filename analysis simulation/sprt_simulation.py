import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed, cpu_count
from matplotlib2tikz import save as tikz_save
from sklearn import svm
import pickle
import os


class test():
    def __init__(self, params, number_of_processes=1):
        self.params = params
        self.params['l'] = np.log((1 - self.params['beta']) / self.params['alpha'])
        self.params['h'] = np.log(self.params['beta'] / (1 - self.params['alpha']))
        self.params['Av'] = np.log((self.params['theta_1v'] * (1 - self.params['theta_0v'])) /
                                   (self.params['theta_0v'] * (1 - self.params['theta_1v'])))
        self.params['Bv'] = np.log((1 - self.params['theta_1v']) / (1 - self.params['theta_0v']))
        self.nop = number_of_processes
        # Initialize parameters of defense and attack mechanisms (to be filled after!)
        self.com_p = None
        self.delta = 0

        self.svm = None
        self.window_size = 5  # Window to autocorrelate
        self.sample_size = 2 * self.window_size - 1  # Size of the autocorrelation vector
        self.tr_size = 500
        self.val_size = 500

    def features_extraction(self, x):  # Used to extract the features for the 1 class SVM
        x -= np.mean(x)
        x = np.abs(np.fft.fft(np.correlate(x, x, 'full')))
        return x / np.amax(x)

    def train_svm(self, theta):
        # Get the SVM with lower error!
        tryouts = 5
        self.svm = None  # Delete any previous SVM
        svm_vector = []
        training_error = np.zeros(tryouts)
        pred_error = np.zeros(tryouts)
        for isvm in range(tryouts):
            svm_vector.append(svm.OneClassSVM(nu=0.1, gamma='auto'))  # Create svm
            # Train the SVM
            x = np.random.binomial(1, theta, size=self.tr_size)
            t = np.arange(len(x))
            x = np.cumsum(x) * self.com_p['A'] + (t + 1) * self.com_p['B']
            number_of_seqs = len(x) - self.window_size
            batch_train = np.zeros([number_of_seqs, self.sample_size])
            for i in range(number_of_seqs):
                batch_train[i, :] = self.features_extraction(x[i: i + self.window_size])
            svm_vector[isvm].fit(batch_train)
            y_pred_train = svm_vector[isvm].predict(batch_train)

            # Validation
            x_val = np.random.binomial(1, theta, size=self.val_size)
            t = np.arange(len(x_val))
            x_val = np.cumsum(x_val) * self.com_p['A'] + (t + 1) * self.com_p['B']
            number_of_seqs = len(x_val) - self.window_size
            batch_val = np.zeros([number_of_seqs, self.sample_size])
            for i in range(number_of_seqs):
                batch_val[i, :] = self.features_extraction(x_val[i: i + self.window_size])
            y_pred_val = svm_vector[isvm].predict(batch_val)

            training_error[isvm] = y_pred_train[y_pred_train == -1].size / len(x)
            pred_error[isvm] = y_pred_val[y_pred_val == -1].size / len(x_val)
        # SEt as SVM the one with lower error
        self.svm = svm_vector[np.argmin(training_error + pred_error)]

    def obtain_test(self, mode_attack='no_attack', mode_defense='sprt', delta = 0, save_fig=False):
        self.delta = delta  # Initialize delta velue
        def sprt_test(x_n, sim, it, result, length, reward, lock, correlator):
            self.com_p['s'] += x_n
            LLR_n = self.com_p['LLR'][-1]
            if x_n == 0:
                LLR_n += self.com_p['B']
            else:
                LLR_n += self.com_p['A'] + self.com_p['B']
            if mode_defense == 'sprt_svm':
                LLR_n += correlator * self.delta * (self.com_p['A'] + self.com_p['B'])
            self.com_p['LLR'].append(LLR_n)
            reward[sim, self.com_p['n'], it] = x_n
            # if (LLR_n >= self.params['l'] or correlator == 1) and not self.com_p['test_done']:
            if LLR_n >= self.params['l'] and not self.com_p['test_done']:
                result[sim, it] = 1  # H0 rejected
                length[sim, it] = self.com_p['n'] + 1
                self.com_p['n'] = self.params['nmax']  # This is to break loop if attacker is detected!!
                self.com_p['test_done'] = True
            elif LLR_n <= self.params['h'] and not self.com_p['test_done']:
                result[sim, it] = 0  # H0 accepted
                length[sim, it] = self.com_p['n'] + 1
                self.com_p['n'] += 1  # If no attacker detected, keep on communicating!
                self.com_p['test_done'] = True
            elif self.com_p['n'] == self.params['nmax'] - 1 and not self.com_p['test_done']:
                result[sim, it] = 2  # No decision taken
                length[sim, it] = self.com_p['n'] + 1
                self.com_p['n'] += 1
            else:
                self.com_p['n'] += 1
            if self.com_p['n'] >= self.params['nlock'] and lock:
                self.com_p['test_done'] = True
            return result, length, reward

        result = -np.ones([self.params['nsim'], len(self.params['theta_0v'])])
        length = -np.ones([self.params['nsim'], len(self.params['theta_0v'])])
        reward = np.zeros([self.params['nsim'], self.params['nmax'], len(self.params['theta_0v'])])

        for it in range(len(self.params['theta_0v'])):
            # Clear parameters
            self.com_p = None
            theta_0 = self.params['theta_0v'][it]
            theta_1 = self.params['theta_1v'][it]
            lock = False
            if mode_attack == 'attack_lock':
                lock = True
            correlator = 0  # Correlator test result (default: do not use it)

            # Common parameters
            self.com_p = {'n': 0, 's': 0, 'test_done': False, 'A': self.params['Av'][it], 'B': self.params['Bv'][it]}

            if mode_defense == 'sprt_svm':
                self.train_svm(theta_0)

            for sim in range(self.params['nsim']):
                acs = []
                self.com_p['n'] = 0
                self.com_p['s'] = 0
                self.com_p['test_done'] = False
                self.com_p['LLR'] = [0]  # LLR will be updated in the loop
                if mode_attack == 'no_attack':
                    x = np.random.binomial(1, theta_0, size=self.params['nmax'])  # Generate data
                elif mode_attack == 'dummy_attack':
                    x = np.random.binomial(1, theta_1, size=self.params['nmax'])  # Generate data
                # TEST!!
                while self.com_p['n'] < self.params['nmax']:
                    # Take action
                    if mode_attack == 'no_attack' or mode_attack == 'dummy_attack':
                        x_n = x[self.com_p['n']]
                    elif mode_attack == 'attack_no_lock':
                        g1 = (self.com_p['s'] + 1) * self.com_p['A'] + (self.com_p['n'] + 1) * self.com_p['B']
                        if g1 < self.params['l']:
                            x_n = 1
                        else:
                            x_n = 0
                    elif mode_attack == 'attack_lock':
                        g1 = (self.com_p['s'] + 1) * self.com_p['A'] + (self.com_p['n'] + 1) * self.com_p['B']
                        glock = self.com_p['s'] * self.com_p['A'] + (self.com_p['n'] + 1) * self.com_p['B'] \
                                + self.com_p['A'] + self.com_p['B'] * (self.params['nlock'] - self.com_p['n'] - 1)
                        if g1 < self.params['l'] and glock <= self.params['h']:
                            x_n = 1
                        elif self.com_p['test_done']:
                            x_n = 1
                        else:
                            x_n = 0
                    else:
                        raise NotImplementedError
                    # Store actions!
                    acs.append(x_n)
                    # Defense mechanism
                    if mode_defense == 'sprt_svm' and not self.com_p['test_done']:
                        if len(acs) < self.window_size:
                            correlator = 0
                        else:
                            tv = np.array(acs[-self.window_size:])
                            t = np.arange(self.com_p['n'] - self.window_size, self.com_p['n'])
                            tv = np.cumsum(tv) * self.com_p['A'] + (t + 1) * self.com_p['B']
                            correlator = self.svm.predict(self.features_extraction(tv).reshape([1, self.sample_size]))
                            if correlator == -1:
                                correlator = 1
                            else:
                                correlator = 0
                    if mode_defense == 'dist':
                        expected = theta_0 * self.com_p['n'] * self.com_p['A'] + (self.com_p['n'] + 1) * self.com_p['B']
                        actual = self.com_p['s'] * self.com_p['A'] + (self.com_p['n'] + 1) * self.com_p['B']
                        if actual >= expected + 2 * self.params['l']:
                            correlator = 1
                        else:
                            correlator = 0

                    result, length, reward = sprt_test(x_n, sim, it, result, length, reward, lock, correlator)

                if save_fig:
                    # Plot data
                    #if mode == 'attack_lock':
                    #    acs = acs[0: int(self.params['nlock'] * 1.2)]
                    t = np.arange(len(acs))
                    plt.plot(t, np.ones_like(t) * self.params['l'], 'b')
                    plt.plot(t, np.ones_like(t) * self.params['h'], 'b')
                    acs = np.array(acs)
                    plt.plot(t, self.com_p['LLR'][1:], 'ro-')
                    plt.axvline(x=length[sim, it] - 1, color='g')
                    plt.title(mode_attack + ' - ' + mode_defense + ' Theta_0 = ' +str(theta_0) + ' theta_1 = ' + str(theta_1))
                    tikz_save(mode_attack + '_' + mode_defense + '.tikz',
                              figureheight='\\figureheight', figurewidth='\\figurewidth')
                    plt.show()

            n_of_h0 = np.sum(result[:, it] == 0)
            n_of_h1 = np.sum(result[:, it] == 1)
            n_of_lock = np.sum(result[:, it] == 2)
            print(mode_attack + ' - ' + mode_defense + ' - ' + str(self.delta),
                  'Theta_0 = ', theta_0, ' theta_1 = ', theta_1,
                  '; H1 = ', n_of_h1 / self.params['nsim'],
                  '; H0 = ', n_of_h0 / self.params['nsim'],
                  '; Lock = ', n_of_lock / self.params['nsim'])
        return result, length, reward



if __name__ == '__main__':

    #Check for saved values

    if not os.path.isfile(os.path.normpath(os.getcwd() + '/params.pickle')):

        # Test data
        params = {'nsim' : 500, # Number of simulations to average
                  'alpha' : 0.05,
                  'beta' : 0.05,
                  'nmax' : 200,  # Max length of simulation
                  'nlock' : 100,  # Max lock time
                  'theta_0v' : np.linspace(0.1, 0.7, 10),
                  'theta_1v' : np.linspace(0.3, 0.9, 10),
                  'ata_m' : ['no_attack', 'dummy_attack', 'attack_no_lock', 'attack_lock'],
                  'delta_v' : [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1],
                  'delta':  0.05
                  }


        t = test(params)

        results = []
        num_cores = 7  # Multiprocessing

        # Simulate without advanced defense mechanism

        def proc1(at):
            res, le, rew = t.obtain_test(mode_attack=at, mode_defense='sprt')
            data = {'res': res, 'len': le, 'rew': rew, 'at': at, 'def': 'sprt', 'delta': None}
            return data


        out = Parallel(n_jobs=num_cores, verbose=5)(delayed(proc1)(at=at) for at in params['ata_m'])
        results.extend(out)

        # Simulate with advanced defense mechanism and fixed delta
        def proc2(at):
            res, le, rew = t.obtain_test(mode_attack=at, mode_defense='sprt_svm', delta=params['delta'])
            data = {'res': res, 'len': le, 'rew': rew, 'at': at, 'def': 'sprt_svm', 'delta': params['delta']}
            return data

        out = Parallel(n_jobs=num_cores, verbose=5)(delayed(proc2)(at=at) for at in params['ata_m'])
        results.extend(out)

        # Simulate to infer what happens when using delta
        def proc3(at, delta):
            res, le, rew = t.obtain_test(mode_attack=at, mode_defense='sprt_svm', delta=delta)
            data = {'res': res, 'len': le, 'rew': rew, 'at': at, 'def': 'sprt_svm', 'delta': delta}
            return data

        out = Parallel(n_jobs=num_cores, verbose=5)(delayed(proc3)(at=at, delta=delta)
                                                    for at in ['no_attack', 'dummy_attack']
                                                    for delta in params['delta_v'])
        results.extend(out)

        # Save data
        with open('params.pickle', 'wb') as handle:
            pickle.dump(params, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open('results.pickle', 'wb') as handle:
            pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
    else:  #Load saved parameters
        with open('params.pickle', 'rb') as handle:
            params = pickle.load(handle)
        with open('results.pickle', 'rb') as handle:
            results = pickle.load(handle)


    # Results extraction
    ndf = 30
    disc_factors = np.linspace(0.01, 0.99, ndf)
    disc_matrix = np.zeros([ndf, params['nmax']])
    for df in range(ndf):
        disc_matrix[df, :] = np.power(disc_factors[df], np.arange(start=0, stop=params['nmax']))

    def obtain_output_values(data):
        if np.sum(data['len'] == -1) > 0 or np.sum(data['res'] == -1) > 0:
            raise RuntimeError('Invalid values found')
        x_axis = data['res'].shape[1]
        length = np.mean(data['len'], axis=0)
        te = np.zeros([x_axis, 3])
        for i in range(x_axis):
            te[i, 0] = np.sum(data['res'][:, i] == 0) / params['nsim']
            te[i, 1] = np.sum(data['res'][:, i] == 1) / params['nsim']
            te[i, 2] = np.sum(data['res'][:, i] == 2) / params['nsim']
        reward = np.zeros([x_axis, ndf])
        for i in range(x_axis):
            for df in range(ndf):
                reward[i, df] = np.sum(np.mean(data['rew'][:, :, i], axis=0) * disc_matrix[df, :])
        return reward, length, te

    r = []
    l = []
    res = []
    lab = []

    for data in results:
        rew, le, te = obtain_output_values(data)
        r.append(rew)
        l.append(le)
        res.append(te)
        lab.append(data['at'] + ' ' + data['def'] + ' ' + str(data['delta']))

    # Plot 0: influence of delta value in SPRT SVM
    x = params['delta_v']
    t0 = 0.5
    ith = np.argmin(np.abs(params['theta_0v'] - t0))  # índex of theta value!
    y = np.zeros([len(x), 2, 3])  # delta x attack case x test result
    for at in ['no_attack', 'dummy_attack']:
        iat = params['ata_m'].index(at)
        for d in x:
            ide = x.index(d)
            i = lab.index(at + ' ' + 'sprt_svm' + ' ' + str(d))
            y[ide, iat, :] = res[i][ith, :]

    f, (ax, ax2) = plt.subplots(2, 1, sharex=True)  # Axis break to see results well
    lb = ['H0', 'H1', 'ND']
    for itest in range(3):
        for at in ['no_attack', 'dummy_attack']:
            iat = params['ata_m'].index(at)
            if iat == 0:
                lines = ['g-o', 'r-s', 'k-^']
            else:
                lines = ['g--o', 'r--s', 'k--^']
            icl = lines[itest]
            ax.plot(x, y[:, iat, itest], icl, alpha=1, markersize=10)
            ax2.plot(x, y[:, iat, itest], icl, alpha=1, markersize=10, label=at + ' ' + lb[itest])
    ax.plot(x, params['alpha'] * np.ones_like(x), 'k:', alpha=1)
    ax2.plot(x, params['alpha'] * np.ones_like(x), 'k:', alpha=1)
    ax.plot(x, 1 - params['beta'] * np.ones_like(x), 'k:', alpha=1)
    ax2.plot(x, 1 - params['beta'] * np.ones_like(x), 'k:', alpha=1)
    # Axis break
    ax.set_ylim(0.85, 1.01)
    ax2.set_ylim(-0.01, 0.15)
    ax.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax.xaxis.tick_top()
    ax.tick_params(labeltop=False)  # don't put tick labels at the top
    ax2.xaxis.tick_bottom()
    d = .015  # how big to make the diagonal lines in axes coordinates
    # arguments to pass to plot, just so we don't keep repeating them
    kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
    ax.plot((-d, +d), (-d, +d), **kwargs)  # top-left diagonal
    ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal
    kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
    ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal
    # Final stuff
    plt.xlabel('Delta')
    # plt.ylabel('Probability')
    ax2.legend(loc='best')
    tikz_save('fig_delta.tikz', figureheight='\\figureheight', figurewidth='\\figurewidth')
    plt.show()

    # Plot 1: detection for each defense mechanism
    def_v = ['sprt', 'sprt_svm']
    x = params['theta_0v']
    y = np.zeros([len(x), 3, 4, 2])  # Theta x test_result x atack x defense
    for at in params['ata_m']:
        iat = params['ata_m'].index(at)
        for de in def_v:
            ide = def_v.index(de)
            if ide == 0:
                i = lab.index(at + ' ' + de + ' ' + str(None))
            else:
                i = lab.index(at + ' ' + de + ' ' + str(params['delta']))
            y[:, :, iat, ide] = res[i]
    f, (ax, ax2) = plt.subplots(2, 1, sharex=True)  # Axis break to see results well
    for ide in range(len(def_v)):
        if ide == 0:
            lines = ['s-k', '^-g', 'o-b', 'x-r']
        else:
            lines = ['s--k', '^--g', 'o--b', 'x--r']
        for at in params['ata_m']:
            iat = params['ata_m'].index(at)
            icl = lines[iat]
            ax.plot(x, y[:, 1, iat, ide], icl, alpha=1, markersize=10)
            ax2.plot(x, y[:, 1, iat, ide], icl, alpha=1, markersize=10, label=at + ' H1 ' + def_v[ide])
    ax.plot(x, params['alpha'] * np.ones_like(x), 'k:', alpha=1)
    ax2.plot(x, params['alpha'] * np.ones_like(x), 'k:', alpha=1)
    ax.plot(x, 1 - params['beta'] * np.ones_like(x), 'k:', alpha=1)
    ax2.plot(x, 1 - params['beta'] * np.ones_like(x), 'k:', alpha=1)
    # Axis break
    ax.set_ylim(0.85, 1.01)
    ax2.set_ylim(-0.01, 0.15)
    ax.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax.xaxis.tick_top()
    ax.tick_params(labeltop=False)  # don't put tick labels at the top
    ax2.xaxis.tick_bottom()
    d = .015  # how big to make the diagonal lines in axes coordinates
    # arguments to pass to plot, just so we don't keep repeating them
    kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
    ax.plot((-d, +d), (-d, +d), **kwargs)  # top-left diagonal
    ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal
    kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
    ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal
    # Final stuff
    plt.xlabel('Theta_0')
    # plt.ylabel('Probability')
    ax2.legend(loc='best')
    tikz_save('fig_h1.tikz', figureheight='\\figureheight', figurewidth='\\figurewidth')
    plt.show()

    # Plot 2: reward under different attacks and defense mechanisms, as a function of discount factor
    plt.xscale('logit')
    def_v = ['sprt', 'sprt_svm']
    x = disc_factors
    t0 = 0.5
    ith = np.argmin(np.abs(params['theta_0v'] - t0))  # índex of theta value!
    y = np.zeros([len(x), 4, 2])  # df x atack x defense
    for at in params['ata_m']:
        iat = params['ata_m'].index(at)
        for de in def_v:
            ide = def_v.index(de)
            if ide == 0:
                i = lab.index(at + ' ' + de + ' ' + str(None))
            else:
                i = lab.index(at + ' ' + de + ' ' + str(params['delta']))
            y[:, iat, ide] = r[i][ith, :].flatten()

    for ide in range(len(def_v)):
        if ide == 0:
            lines = ['-k', '-g', '-b', '-r']
        else:
            lines = ['--k', '--g', '--b', '--r']
        for at in params['ata_m']:
            iat = params['ata_m'].index(at)
            icl = lines[iat]
            plt.semilogy(x, y[:, iat, ide], icl, alpha=1, label=at + def_v[ide])
    # Final stuff
    plt.xlabel('Disc factor')
    plt.ylabel('R')
    plt.legend(loc='best')
    plt.title('Rewards evolution for theta_0 = ' + str(params['delta']))
    tikz_save('fig_rew.tikz', figureheight='\\figureheight', figurewidth='\\figurewidth')
    plt.show()

    # Performance table example
    t0 = 0.5
    ith = np.argmin(np.abs(params['theta_0v'] - t0))  # índex of theta value!

    print('Performance for theta_0 = ', t0)

    for at in params['ata_m']:
        iat = params['ata_m'].index(at)
        for de in def_v:
            ide = def_v.index(de)
            if ide == 0:
                i = lab.index(at + ' ' + de + ' ' + str(None))
            else:
                i = lab.index(at + ' ' + de + ' ' + str(params['delta']))
            print(at + ' ' + de + ' ' + str(res[i][ith, :]))

    # Examples of each SPRT threshold
    # Modify the params!
    params['nsim'] = 1
    params['theta_0v'] = np.array([0.5])
    params['theta_1v'] = np.array([0.7])
    t = test(params)
    for at in params['ata_m']:
        _, _, _ = t.obtain_test(mode_attack=at, mode_defense='sprt', save_fig=True)
        _, _, _ = t.obtain_test(mode_attack=at, mode_defense='sprt_svm', delta=params['delta'], save_fig=True)