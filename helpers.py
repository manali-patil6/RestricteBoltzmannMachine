import numpy as np
import torch


def convert(data, nb_users, nb_movies):
    new_data = []
    for id_users in range(1, nb_users + 1):
        id_movies = data[:, 1][data[:, 0] == id_users]
        id_ratings = data[:, 2][data[:, 0] == id_users]
        ratings = np.zeros(nb_movies)
        ratings[id_movies - 1] = id_ratings
        new_data.append(list(ratings))
    return new_data


def trainer(nb_epoch, nb_users, batch_size, training_set, rbm):
    for epoch in range(1, nb_epoch + 1):
        train_loss = 0
        s = 0.
        for id_user in range(0, nb_users - batch_size, batch_size):
            vk = training_set[id_user: id_user + batch_size]
            v0 = training_set[id_user: id_user + batch_size]
            ph0, _ = rbm.sample_h(v0)
            for k in range(10):
                _, hk = rbm.sample_h(vk)
                _, vk = rbm.sample_v(hk)
                vk[v0 < 0] = v0[v0 < 0]
            phk, _ = rbm.sample_h(vk)
            rbm.train(v0, vk, ph0, phk)
            train_loss += torch.mean(torch.abs(v0[v0 >= 0] - vk[v0 >= 0]))
            s += 1.
        print('epoch: ' + str(epoch) + ' loss: ' + str(train_loss / s))
        return str(train_loss / s)


def tester(nb_users, test_set, training_set, rbm):
    test_loss = 0
    s = 0.
    for id_user in range(nb_users):
        v = training_set[id_user:id_user + 1]
        vt = test_set[id_user:id_user + 1]
        if len(vt[vt >= 0]) > 0:
            _, h = rbm.sample_h(v)
            _, v = rbm.sample_v(h)
            test_loss += torch.mean(torch.abs(vt[vt >= 0] - v[vt >= 0]))
            s += 1.
    print('test loss: ' + str(test_loss / s))
    return str(test_loss / s)
