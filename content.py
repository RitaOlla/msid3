# --------------------------------------------------------------------------
# ------------  Metody Systemowe i Decyzyjne w Informatyce  ----------------
# --------------------------------------------------------------------------
#  Zadanie 3: Regresja logistyczna
#  autorzy: A. Gonczarek, J. Kaczmar, S. Zareba
#  2017
# --------------------------------------------------------------------------
import functools

import numpy as np


def sigmoid(x):
    """
    :param x: wektor wejsciowych wartosci Nx1
    :return: wektor wyjściowych wartości funkcji sigmoidalnej dla wejścia x, Nx1
    """
    return 1 / (1 + np.exp(-x))

def logistic_cost_function(w, x_train, y_train):
    """
    :param w: parametry modelu Mx1
    :param x_train: ciag treningowy - wejscia NxM
    :param y_train: ciag treningowy - wyjscia Nx1
    :return: funkcja zwraca krotke (val, grad), gdzie val oznacza wartosc funkcji logistycznej, a grad jej gradient po w
    """
    N = np.shape(x_train)[0]
    M = np.shape(x_train)[1]
    y_trainT = np.transpose(y_train)
    wtx = x_train @ w
    s = sigmoid(wtx)
    # 60ms przy 5000 probach
    a = y_trainT @ np.log(s) + (1 - y_trainT) @ np.log(1 - s)
    val = (-1 / N) * a[0, 0]


    t = s - y_train
    grad = (-1 / N) * (np.transpose(-x_train) @ t)

    return val, grad

def gradient_descent(obj_fun, w0, epochs, eta):
    """
    :param obj_fun: funkcja celu, ktora ma byc optymalizowana. Wywolanie val,grad = obj_fun(w).
    :param w0: punkt startowy Mx1
    :param epochs: liczba epok / iteracji algorytmu
    :param eta: krok uczenia
    :return: funkcja wykonuje optymalizacje metoda gradientu prostego dla funkcji obj_fun. Zwraca krotke (w,func_values),
    gdzie w oznacza znaleziony optymalny punkt w, a func_valus jest wektorem wartosci funkcji [epochs x 1] we wszystkich krokach algorytmu
    """
    w = np.copy(w0)
    func_vals = np.zeros(shape=(epochs, 1))
    for i in range(epochs):
        func_vals[i - 1, 0], grad = obj_fun(w)
        w += eta * -grad

    func_vals[epochs - 1, 0], _ = obj_fun(w)

    return w, func_vals

def stochastic_gradient_descent(obj_fun, x_train, y_train, w0, epochs, eta, mini_batch):
    """
    :param obj_fun: funkcja celu, ktora ma byc optymalizowana. Wywolanie val,grad = obj_fun(w,x,y), gdzie x,y oznaczaja podane
    podzbiory zbioru treningowego (mini-batche)
    :param x_train: dane treningowe wejsciowe NxM
    :param y_train: dane treningowe wyjsciowe Nx1
    :param w0: punkt startowy Mx1
    :param epochs: liczba epok
    :param eta: krok uczenia
    :param mini_batch: wielkosc mini-batcha
    :return: funkcja wykonuje optymalizacje metoda stochastycznego gradientu prostego dla funkcji obj_fun. Zwraca krotke (w,func_values),
    gdzie w oznacza znaleziony optymalny punkt w, a func_values jest wektorem wartosci funkcji [epochs x 1] we wszystkich krokach algorytmu. Wartosci
    funkcji do func_values sa wyliczane dla calego zbioru treningowego!
    """
    N = np.shape(y_train)[0]
    MB_size = np.floor(N / mini_batch)

    w = np.copy(w0)
    func_vals = np.zeros(shape=(epochs, 1))
    batchesX = np.array_split(x_train, MB_size, axis=0)
    batchesY = np.array_split(y_train, MB_size, axis=0)
    for i in range(epochs):
        val, _ = obj_fun(w, x_train, y_train)
        func_vals[i - 1, 0] = val
        for m in range(np.shape(batchesX)[0]):
            _, grad = obj_fun(w, batchesX[m], batchesY[m])
            w += eta * -grad

    func_vals[epochs - 1, 0], _ = obj_fun(w, x_train, y_train)

    return w, func_vals

def regularized_logistic_cost_function(w, x_train, y_train, regularization_lambda):
    """
    :param w: parametry modelu Mx1
    :param x_train: ciag treningowy - wejscia NxM
    :param y_train: ciag treningowy - wyjscia Nx1
    :param regularization_lambda: parametr regularyzacji
    :return: funkcja zwraca krotke (val, grad), gdzie val oznacza wartosc funkcji logistycznej z regularyzacja l2,
    a grad jej gradient po w
    """
    val, grad = logistic_cost_function(w, x_train, y_train)
    w_0 = np.copy(w)
    w_0[0] = 0

    val += regularization_lambda * 0.5 * np.sum(w_0 * w_0)
    grad += regularization_lambda * w_0

    return val, grad

def prediction(x, w, theta):
    """
    :param x: macierz obserwacji NxM
    :param w: wektor parametrow modelu Mx1
    :param theta: prog klasyfikacji z przedzialu [0,1]
    :return: funkcja wylicza wektor y o wymiarach Nx1. Wektor zawiera wartosci etykiet ze zbioru {0,1} dla obserwacji z x
     bazujac na modelu z parametrami w oraz progu klasyfikacji theta
    """
    wtx = x @ w
    s = sigmoid(wtx)
    return s >= theta

def f_measure(y_true, y_pred):
    """
    :param y_true: wektor rzeczywistych etykiet Nx1
    :param y_pred: wektor etykiet przewidzianych przed model Nx1
    :return: funkcja wylicza wartosc miary F
    """
    a = y_true * y_pred
    b = y_true * np.logical_not(y_pred)
    c = np.logical_not(y_true) * y_pred

    tp = np.count_nonzero(a)
    fp = np.count_nonzero(b)
    fn = np.count_nonzero(c)

    return (2 * tp) / (2 * tp + fp + fn)

def model_selection(x_train, y_train, x_val, y_val, w0, epochs, eta, mini_batch, lambdas, thetas):
    """
    :param x_train: ciag treningowy wejsciowy NxM
    :param y_train: ciag treningowy wyjsciowy Nx1
    :param x_val: ciag walidacyjny wejsciowy Nval x M
    :param y_val: ciag walidacyjny wyjsciowy Nval x 1
    :param w0: wektor poczatkowych wartosci parametrow
    :param epochs: liczba epok dla SGD
    :param eta: krok uczenia
    :param mini_batch: wielkosc mini batcha
    :param lambdas: lista wartosci parametru regularyzacji lambda, ktore maja byc sprawdzone
    :param thetas: lista wartosci progow klasyfikacji theta, ktore maja byc sprawdzone
    :return: funckja wykonuje selekcje modelu. Zwraca krotke (regularization_lambda, theta, w, F), gdzie regularization_lambda
    to najlpszy parametr regularyzacji, theta to najlepszy prog klasyfikacji, a w to najlepszy wektor parametrow modelu.
    Dodatkowo funkcja zwraca macierz F, ktora zawiera wartosci miary F dla wszystkich par (lambda, theta). Do uczenia nalezy
    korzystac z algorytmu SGD oraz kryterium uczenia z regularyzacja l2.
    """
    lam_max = lambdas[0]
    theta_max = thetas[0]
    w_best = None
    fm_max = None
    F = np.zeros(shape=(np.size(lambdas), np.size(thetas)))

    for i in range(np.size(lambdas)):
        lam = lambdas[i]
        w, _ = stochastic_gradient_descent(
            functools.partial(regularized_logistic_cost_function, regularization_lambda=lam),
            x_train, y_train, w0, epochs, eta, mini_batch
        )
        for j in range(np.size(thetas)):
            theta = thetas[j]
            y_pred = prediction(x_val, w, theta)
            fm = f_measure(y_val, y_pred)
            F[i, j] = fm
            if fm_max is None or fm > fm_max:
                lam_max = lam
                theta_max = theta
                w_best = w
                fm_max = fm

    return lam_max, theta_max, w_best, F
