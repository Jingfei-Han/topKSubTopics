from keras.models import Sequential
from keras.models import load_model
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD
import keras
import numpy as np
import os
import pickle

from globalVar import w2v_model, taxonomy

class MLP(object):
    def __init__(self, emb = 200, nb_input=3, filename="mlp_model.h5", isReadModel=False, isRun=False):
        self.model = Sequential()
        #self.train_X, self.train_y, self.test_X, self.test_y = [], [], [], []
        self.emb = emb
        self.nb_input = nb_input
        self.input_dim = self.nb_input * self.emb
        self.output_dim = 1

        self.batch_size = 64

        self.filename = filename
        self.isReadModel = isReadModel
        self.isRun = isRun


    def build_model(self):
        self.model.add(Dense(output_dim=30, input_shape=(self.input_dim, )))
        self.model.add(Activation("linear"))
        self.model.add(Dense(output_dim=30))
        self.model.add(Activation("linear"))
        self.model.add(Dense(output_dim=self.output_dim))
        self.model.add(Activation("sigmoid"))

        self.model.summary()

        self.model.compile(loss='mean_squared_error', optimizer=SGD(), metrics=[keras.metrics.mse, keras.metrics.mae])

    def fit(self, train_X, train_y, test_X, test_y, epoch=20):
        self.model.fit(train_X, train_y, validation_data=(test_X, test_y),
                       epochs=epoch, batch_size=self.batch_size, verbose=True)

    def eval(self, test_X, test_y):
        loss_and_metrics = self.model.evaluate(test_X, test_y, verbose=False)
        return loss_and_metrics

    def _predict(self, test_X):
        test_y_hat = self.model.predict(test_X)
        return test_y_hat

    def train_model(self, emb=200, epoch=20):
        if not self.isRun:
            if self.isReadModel and os.path.exists(self.filename):
                print("read model...")
                mlp_model = load_model(self.filename)
                self.model = mlp_model
                print("read model finished.")
            else:
                print("start to train model...")
                train_X, train_y, test_X, test_y = generate_data(alpha = 0.7)
                self.build_model()
                if os.path.exists(self.filename):
                    print("pre-train model using file...")
                    self.model = load_model(self.filename)
                self.fit(train_X, train_y, test_X, test_y, epoch=epoch)
                loss_and_metrics = self.eval(test_X, test_y)
                print("loss and metrics is: ", loss_and_metrics)
                print("Train model finished.")

                self.save_model() #save model when training finished.
        else:
            self.model = load_model(self.filename)

    def test_model(self, parent, children, context):
        try:
            tmp_X = np.asarray([w2v_model[children], w2v_model[parent], w2v_model[context]])
            tmp_X = tmp_X.reshape(1, -1)
            y_hat = self._predict(tmp_X)
            return y_hat
        except:
            return 0

    def predict(self, area, children, context="computer_science"):
        emb_A = w2v_model[area]
        emb_B = []
        emb_context = w2v_model[context]
        embSize = len(emb_context)

        resList = []
        for i in children:
            try:
                emb_B.append(w2v_model[i])
                resList.append(i)
            except:
                pass

        cnt = len(resList)
        assert len(emb_B) == cnt

        emb_context = np.repeat(emb_context.reshape(1, -1), cnt, axis=0)
        emb_A = np.repeat(emb_A.reshape(1, -1), cnt, axis=0)

        data = np.concatenate((emb_B, emb_A, emb_context), axis=1)
        #print(data)
        #print(data.shape)
        assert data.shape[1] == 3 * embSize

        y_hat = self._predict(data)
        y_hat = [float(i) for i in y_hat]
        #print(y_hat)
        #print(resList)
        res = dict(zip(resList, y_hat))
        return res

    def save_model(self):
        print("save model...")
        self.model.save(self.filename)
        print("The model was saved.")
    """
    def load_model(self, ):
        self.model = load_model(filename)
    """
def generate_data(alpha = 0.9):
    print("---------------------------------------")
    print("start to generate data...")
    cnt = 0
    maxCnt = 1000000 # max count
    #context = "computer_science"
    #contextList = ["physic", "math", "chemistry", "biology", "engineering", "biochemistry", "geography",
    #               "linguistics", "philosophy", "computer_science"]
    contextList = ["physic", "math", "chemistry", "computer_science"]
    #contextList = ["computer_science"]
    #emb_context = w2v_model[context]
    # emb_A: parent, emb_B: children, emb_context: context
    emb_A = []
    emb_B = []
    emb_context = []
    y = []
    y_name = []
    emb_A_cs = [] #computer_science
    emb_B_cs = []
    emb_context_cs = []
    y_cs = []
    y_name_cs = []
    for (i, j) in taxonomy.items():
        for k in j['subcats']:
            for context in contextList:
                try:
                    tmp_score = w2v_model.n_similarity([k], [i, context])
                    #if i == "machine_learning":
                        #print(i, k, context, tmp_score)
                except:
                    continue
                if tmp_score > 0.3:
                    if context == "computer_science":
                        emb_A_cs.append(w2v_model[i])
                        emb_B_cs.append(w2v_model[k])
                        emb_context_cs.append(w2v_model[context])
                        y_cs.append(tmp_score)
                        y_name_cs.append(k + " : " + i + " : " + context)
                    else:
                        emb_A.append(w2v_model[i])
                        emb_B.append(w2v_model[k])
                        emb_context.append(w2v_model[context])
                        y.append(tmp_score)
                        y_name.append(k + " : " + i + " : " + context)

                    cnt += 1
                if cnt > maxCnt:
                    break
            if cnt > maxCnt:
                break
        if cnt > maxCnt:
            break

    cnt = len(y)
    embSize = len(emb_context[0])
    emb_context = np.asarray(emb_context)
    emb_A = np.asarray(emb_A)
    emb_B = np.asarray(emb_B)
    y = np.asarray(y).reshape(-1, 1)
    # emb_context, emb_A, emb_B's dim = cnt * embSize = cnt * 200

    cnt_cs = len(y_cs)
    emb_context_cs = np.asarray(emb_context_cs)
    emb_A_cs = np.asarray(emb_A_cs)
    emb_B_cs = np.asarray(emb_B_cs)
    y_cs = np.asarray(y_cs).reshape(-1, 1)

    y_name = np.asarray(y_name).reshape(-1, 1)
    y_name_cs = np.asarray(y_name_cs).reshape(-1, 1)

    data = np.concatenate((emb_B, emb_A, emb_context), axis=1)
    data_cs = np.concatenate((emb_B_cs, emb_A_cs, emb_context_cs), axis=1)
    assert data.shape[1] == 3 * embSize
    assert data_cs.shape[1] == 3 * embSize
    #shuffle and divide data into train and test set

    #trainSize = int(alpha * cnt)
    trainSize = cnt

    print("Sample count: ", cnt+cnt_cs)
    print("Train size: ", trainSize)
    print("Test size: ", cnt_cs)
    print("---------------------------------------")
    rm = np.random.permutation(cnt)
    rm_test = np.random.permutation(cnt_cs)
    """
    train_X = data[rm[:trainSize], :]
    train_y = y[rm[:trainSize], :]
    test_X = data[rm[trainSize:], :]
    test_y = y[rm[trainSize:], :]
    """
    train_X = data[rm, :]
    train_y = y[rm, :]
    test_X = data_cs[rm_test, :]
    test_y = y_cs[rm_test, :]

    train_y_name = y_name[rm, :]
    test_y_name = y_name_cs[rm_test, :]

    print("train sample case study:")
    print(train_y_name[:5], train_y[:5])
    print(test_y_name[:5], test_y[:5])
    #return emb_context, emb_A, emb_B, y
    print("Data generation is finished!")
    print("---------------------------------------")

    #generate pickle file
    with open("dataset.pkl", "wb") as f1:
        pickle.dump((train_X, train_y, test_X, test_y), f1, True)


    return train_X, train_y, test_X, test_y


if __name__ == "__main__":
    model = MLP()
    model.train_model(epoch=2000)
