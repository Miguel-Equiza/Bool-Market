import numpy as np
import pandas as pd
import scipy
import plotly.graph_objects as go

def num(n):
    num_2 = np.random.randint(50,n/3+1)
    num_1 = np.random.randint(1,n-num_2-1)
    num_3 = n-num_2-num_1
    return num_1, num_2, num_3

def X_sart_end(num_1, num_2, num_3):
    X = pd.to_datetime(np.arange(num_1+num_2+num_3), unit='D',
        origin=pd.Timestamp('2017-05-08'))
    start = num_1
    end = num_1+num_2
    return X, start, end

def create_ar2_norm_open(num_1, num_3, df_open2, mu, sigma, ar1, ar2):
    y0=0
    df_open1=[]
    df_open3=[]
    for i in range(num_1):

        if i == 0:
            df_open1.append(0)
            y1 = df_open1[-1]
        else:

            df_open1.append(ar1 * y1 + ar2* y0 + scipy.stats.norm.rvs(mu,sigma))
            y0=y1
            y1 = df_open1[-1]

    for i in range(num_3):
        if i == 0:
            df_open3.append(0)
            y1 = df_open3[-1]
        else:

            df_open3.append(ar1 * y1 + ar2* y0 + scipy.stats.norm.rvs(mu,sigma))
            y0=y1
            y1 = df_open3[-1]


    df_open1.reverse()
    df_open1 = list(np.array(df_open1) + df_open2[0])
    df_open3 = list(np.array(df_open3) +df_open2[-1])


    return df_open1, df_open3

def create_close(df_open1, df_open2, df_open3):
    df_close1=[]
    df_close2=[]
    df_close3=[]
    for i in range(len(df_open1)):
        if i != (len(df_open1)-1):

            df_close1.append(df_open1[i+1])
        else:

            df_close1.append(df_open1[i])

    for i in range(len(df_open2)):
        if i != (len(df_open2)-1):
            df_close2.append(df_open2[i+1])

        else:

            df_close2.append(df_open2[i])

    for i in range(len(df_open3)):
        if i != (len(df_open3)-1):

            df_close3.append(df_open3[i+1])
        else:

            df_close3.append(df_open3[i])
    return df_close1, df_close2, df_close3

def create_low_high(df_open1, df_close1, df_open2, df_close2, df_open3, df_close3, h):
    df_low1=[]
    df_high1=[]
    df_low2=[]
    df_high2=[]
    df_low3=[]
    df_high3=[]
    for i in range(len(df_open1)):

        df_low1.append(min([df_close1[i],df_open1[i]]) - np.random.uniform(0,h))
        df_high1.append(max([df_close1[i],df_open1[i]]) + np.random.uniform(0,h))
    for i in range(len(df_open2)):

        df_low2.append(min([df_close2[i],df_open2[i]]) - np.random.uniform(0,h))
        df_high2.append(max([df_close2[i],df_open2[i]]) + np.random.uniform(0,h))
    for i in range(len(df_open3)):

        df_low3.append(min([df_close3[i],df_open3[i]]) - np.random.uniform(0,h))
        df_high3.append(max([df_close3[i],df_open3[i]]) + np.random.uniform(0,h))

    return df_low1, df_high1, df_low2, df_high2, df_low3, df_high3


def fig_plot(date, ope, hig, low, close, start, end):
    fig = go.Figure(data=[go.Candlestick(x=date,
                open=ope,
                high=hig,
                low=low,
                close=close)])
    fig.add_vline(x = date[start])
    fig.add_vline(x = date[end])
    fig.show()

def norm_params(ar1=0.95, ar2=0, diff=0, n=500, mu=0, sigma=1, h=2):


    X = pd.to_datetime(np.arange(n), unit='D',
        origin=pd.Timestamp('2017-05-08'))
    df_low = []
    df_high = []
    df_close = []
    df_open = []

    y0 = 0
    y1 = 0
    for i in range(n):
        df_open.append(ar1 * y1 + ar2* y0 + scipy.stats.norm.rvs(mu,sigma))
        y1 = df_open[-1]
    for i in range(n):
        if i != (n-1):
            df_close.append(df_open[i+1])
        else:
            df_close.append(df_open[i])
    for i in range(n):
        df_low.append(min([df_close[i],df_open[i]]) - np.random.uniform(0,h))
        df_high.append(max([df_close[i],df_open[i]]) + np.random.uniform(0,h))

    return X, df_open, df_high, df_low, df_close, 0, 0


def double_bottom(n=500, mu=3, sigma=5, h=1, noise_level=2, ar1 = 0.95, ar2 = 0):

    num_1, num_2, num_3 = num(n)

    X, start, end = X_sart_end(num_1, num_2, num_3)


    x = np.linspace(-1.4,3.4,num_2)
    noise = np.random.normal(0,noise_level,num_2)
    #y_open2 = x**4 - 4*(x**3)+ x**2 + 7*x - 3

    b = np.random.uniform(3.9,4.1)
    c = np.random.uniform(4,7)
    d = np.random.uniform(1,5)

    y_open2 = x**4 - b*(x**3)+ x**2 + c*x - d
    df_open2 = 4*y_open2+2*noise
    df_open2 = list(df_open2 + np.random.uniform(-25,25))

    df_open1, df_open3 = create_ar2_norm_open(num_1, num_3, df_open2, mu, sigma, ar1, ar2)

    df_close1, df_close2, df_close3 = create_close(df_open1, df_open2, df_open3)


    df_low1, df_high1, df_low2, df_high2, df_low3, df_high3 = create_low_high(df_open1, df_close1, df_open2, df_close2, df_open3, df_close3, h)


    df_open = df_open1+df_open2+df_open3
    df_close = df_close1+df_close2+df_close3
    df_high = df_high1 + df_high2 + df_high3
    df_low = df_low1 + df_low2 + df_low3


    return X, df_open, df_high, df_low, df_close, start, end


def double_top(n=500, mu=3, sigma=5, h=1, noise_level=2, ar1 = 0.95, ar2 = 0):

    num_1, num_2, num_3 = num(n)

    X, start, end = X_sart_end(num_1, num_2, num_3)

    x = np.linspace(-1.4,3.4,num_2)
    noise = np.random.normal(0,noise_level,num_2)
    #y_open2 = x**4 - 4*(x**3)+ x**2 + 7*x - 3

    b = np.random.uniform(3.9,4.1)
    c = np.random.uniform(4,7)
    d = np.random.uniform(1,5)

    y_open2 = -x**4 + b*(x**3)- x**2 - c*x + d
    df_open2 = 4*y_open2+2*noise
    df_open2 = list(df_open2 + np.random.uniform(-25,25))

    df_open1, df_open3 = create_ar2_norm_open(num_1, num_3, df_open2, mu, sigma, ar1, ar2)

    df_close1, df_close2, df_close3 = create_close(df_open1, df_open2, df_open3)

    df_low1, df_high1, df_low2, df_high2, df_low3, df_high3 = create_low_high(df_open1, df_close1, df_open2, df_close2, df_open3, df_close3, h)

    df_open = df_open1+df_open2+df_open3
    df_close = df_close1+df_close2+df_close3
    df_high = df_high1 + df_high2 + df_high3
    df_low = df_low1 + df_low2 + df_low3


    return X, df_open, df_high, df_low, df_close, start, end

def bearish_flag(n=500, mu=3, sigma=5, h=1, noise_level=2, ar1 = 0.95, ar2=0):

    num_1, num_2, num_3 = num(n)

    X, start, end = X_sart_end(num_1, num_2, num_3)

    a = np.random.uniform(2,8)
    x_1 = np.linspace(-10,0,round(num_2/3))
    x_2 = np.linspace(0,20,round((num_2/3)*2))
    noise_1 = np.random.normal(0,noise_level,round(num_2/3))
    noise_2 = np.random.normal(0,(a/4)*noise_level,round((num_2/3)*2))

    #y_open2 = x**4 - 4*(x**3)+ x**2 + 7*x - 3
    m = np.random.uniform(np.sqrt(3),5)
    y_1 = m*abs(x_1)
    y_1 = 4*y_1 + 2*noise_1

    n = np.random.uniform(0.2,0.8)

    y_2 = n*x_2 + 3*np.sin(x_2)
    y_2 = 4*y_2 + 2*noise_2

    df_open2 = list(y_1) + list(y_2)

    df_open2 = list(np.array(df_open2) + np.random.uniform(-25,25))

    df_open1, df_open3 = create_ar2_norm_open(num_1, num_3, df_open2, mu, sigma, ar1, ar2)

    df_close1, df_close2, df_close3 = create_close(df_open1, df_open2, df_open3)

    df_low1, df_high1, df_low2, df_high2, df_low3, df_high3 = create_low_high(df_open1, df_close1, df_open2, df_close2, df_open3, df_close3, h)


    df_open = df_open1+df_open2+df_open3
    df_close = df_close1+df_close2+df_close3
    df_high = df_high1 + df_high2 + df_high3
    df_low = df_low1 + df_low2 + df_low3


    return X, df_open, df_high, df_low, df_close, start, end

def bullish_flag(n=500, mu=3, sigma=5, h=1, noise_level=2, ar1 = 0.95, ar2 = 0):

    num_1, num_2, num_3 = num(n)

    X, start, end = X_sart_end(num_1, num_2, num_3)


    a = np.random.uniform(2,8)
    x_1 = np.linspace(-10,0,round(num_2/3))
    x_2 = np.linspace(-0.5*np.pi,20,round((num_2/3)*2))
    noise_1 = np.random.normal(0,noise_level,round(num_2/3))
    noise_2 = np.random.normal(0,(a/4)*noise_level,round((num_2/3)*2))

    #y_open2 = x**4 - 4*(x**3)+ x**2 + 7*x - 3
    m = np.random.uniform(np.sqrt(3),5)
    y_1 = 4*m*(x_1) + 3*noise_1

    n = np.random.uniform(0.2,0.8)

    y_2 = -n*x_2 - 3*np.sin(x_2)
    y_2 = 4*y_2 + 2*noise_2

    df_open2 = list(y_1) + list(y_2)

    df_open2 = list(np.array(df_open2) + np.random.uniform(-25,25))

    df_open1, df_open3 = create_ar2_norm_open(num_1, num_3, df_open2, mu, sigma, ar1, ar2)

    df_close1, df_close2, df_close3 = create_close(df_open1, df_open2, df_open3)


    df_low1, df_high1, df_low2, df_high2, df_low3, df_high3 = create_low_high(df_open1, df_close1, df_open2, df_close2, df_open3, df_close3, h)

    df_open = df_open1+df_open2+df_open3
    df_close = df_close1+df_close2+df_close3
    df_high = df_high1 + df_high2 + df_high3
    df_low = df_low1 + df_low2 + df_low3

    return X, df_open, df_high, df_low, df_close, start, end


def bullish_pennant(n=500, mu=3, sigma=5, h=1, noise_level=2, ar1 = 0.95, ar2 = 0):

    num_1, num_2, num_3 = num(n)

    X, start, end = X_sart_end(num_1, num_2, num_3)


    x_1 = np.linspace(-10,0,round(num_2/3))
    #x_2 = np.linspace(0,20,round((num_2/3)*2))
    noise_1 = np.random.normal(0,0.7*noise_level,round(num_2/3))
    noise_2 = np.random.normal(0,noise_level,round((num_2/3)*2))

    #y_open2 = x**4 - 4*(x**3)+ x**2 + 7*x - 3
    m = np.random.uniform(2,5)
    y_1 = 4*m*(x_1) + 3*noise_1

    n = np.random.uniform(0.2,0.8)

    x_2=np.linspace(-0.5*np.pi,30,round((num_2/3)*2))
    x2_2 = np.flip(x_2)
    y_2 = -0.1*x_2 - 3*np.sin(x_2)*((x2_2**2)/200)
    y_2 = 2*y_2 + 2*noise_2
    y_2 = y_2 - y_2[0]
    df_open2 = list(y_1) + list(y_2)

    df_open2 = list(np.array(df_open2) + np.random.uniform(-25,25))

    df_open1, df_open3 = create_ar2_norm_open(num_1, num_3, df_open2, mu, sigma, ar1, ar2)

    df_close1, df_close2, df_close3 = create_close(df_open1, df_open2, df_open3)

    df_low1, df_high1, df_low2, df_high2, df_low3, df_high3 = create_low_high(df_open1, df_close1, df_open2, df_close2, df_open3, df_close3, h)

    df_open = df_open1+df_open2+df_open3
    df_close = df_close1+df_close2+df_close3
    df_high = df_high1 + df_high2 + df_high3
    df_low = df_low1 + df_low2 + df_low3


    return X, df_open, df_high, df_low, df_close, start, end

def bearish_pennant(n=500, mu=3, sigma=5, h=1, noise_level=2, ar1 = 0.95, ar2 = 0):

    num_1, num_2, num_3 = num(n)

    X, start, end = X_sart_end(num_1, num_2, num_3)

    x_1 = np.linspace(-10,0,round(num_2/3))
    #x_2 = np.linspace(0,20,round((num_2/3)*2))
    noise_1 = np.random.normal(0,0.7*noise_level,round(num_2/3))
    noise_2 = np.random.normal(0,noise_level,round((num_2/3)*2))

    #y_open2 = x**4 - 4*(x**3)+ x**2 + 7*x - 3
    m = np.random.uniform(2,5)
    y_1 = -4*m*(x_1) - 3*noise_1

    n = np.random.uniform(0.2,0.8)

    x_2=np.linspace(-0.5*np.pi,30,round((num_2/3)*2))
    x2_2 = np.flip(x_2)
    y_2 = 0.1*x_2 + 3*np.sin(x_2)*((x2_2**2)/200)
    y_2 = 2*y_2 + 2*noise_2
    y_2 = y_2 - y_2[0]
    df_open2 = list(y_1) + list(y_2)

    df_open2 = list(np.array(df_open2) + np.random.uniform(-25,25))

    df_open1, df_open3 = create_ar2_norm_open(num_1, num_3, df_open2, mu, sigma, ar1, ar2)

    df_close1, df_close2, df_close3 = create_close(df_open1, df_open2, df_open3)


    df_low1, df_high1, df_low2, df_high2, df_low3, df_high3 = create_low_high(df_open1, df_close1, df_open2, df_close2, df_open3, df_close3, h)

    df_open = df_open1+df_open2+df_open3
    df_close = df_close1+df_close2+df_close3
    df_high = df_high1 + df_high2 + df_high3
    df_low = df_low1 + df_low2 + df_low3


    return X, df_open, df_high, df_low, df_close, start, end

def descending_triangle(n=500, mu=3, sigma=5, h=1, noise_level=2, ar1 = 0.95, ar2 = 0):

    num_1, num_2, num_3 = num(n)

    X, start, end = X_sart_end(num_1, num_2, num_3)


    x_1 = np.linspace(-10,0,round(num_2/3))
    #x_2 = np.linspace(0,20,round((num_2/3)*2))
    noise_1 = np.random.normal(0,0.7*noise_level,round(num_2/3))
    noise_2 = np.random.normal(0,noise_level,round((num_2/3)*2))

    #y_open2 = x**4 - 4*(x**3)+ x**2 + 7*x - 3
    m = np.random.uniform(2,5)
    y_1 = -4*m*(x_1) - 3*noise_1

    n = np.random.uniform(0.2,0.8)

    x_2 =np.linspace(-20,0.5*np.pi,round((num_2/3)*2))
    y_2 = abs((np.sin(x_2)*x_2+x_2)) + 2*noise_2

    df_open2 = list(y_1) + list(y_2)

    df_open2 = list(np.array(df_open2) + np.random.uniform(-25,25))

    df_open1, df_open3 = create_ar2_norm_open(num_1, num_3, df_open2, mu, sigma, ar1, ar2)

    df_close1, df_close2, df_close3 = create_close(df_open1, df_open2, df_open3)


    df_low1, df_high1, df_low2, df_high2, df_low3, df_high3 = create_low_high(df_open1, df_close1, df_open2, df_close2, df_open3, df_close3, h)


    df_open = df_open1+df_open2+df_open3
    df_close = df_close1+df_close2+df_close3
    df_high = df_high1 + df_high2 + df_high3
    df_low = df_low1 + df_low2 + df_low3


    return X, df_open, df_high, df_low, df_close, start, end

def ascending_triangle(n=500, mu=3, sigma=5, h=1, noise_level=2, ar1 = 0.95, ar2 = 0):

    num_1, num_2, num_3 = num(n)

    X, start, end = X_sart_end(num_1, num_2, num_3)

    df_low1 = []
    df_low2 = []
    df_low3 = []
    df_high1 = []
    df_high2 = []
    df_high3 = []
    df_close = []
    df_open1 = []
    df_open3=[]
    df_close1=[]
    df_close2=[]
    df_close3=[]
    y0=0


    x_1 = np.linspace(-10,0,round(num_2/3))
    #x_2 = np.linspace(0,20,round((num_2/3)*2))
    noise_1 = np.random.normal(0,0.7*noise_level,round(num_2/3))
    noise_2 = np.random.normal(0,noise_level,round((num_2/3)*2))

    #y_open2 = x**4 - 4*(x**3)+ x**2 + 7*x - 3
    m = np.random.uniform(2,5)
    y_1 = 4*m*(x_1) - 3*noise_1

    n = np.random.uniform(0.2,0.8)

    x_2 =np.linspace(-20,0.5*np.pi,round((num_2/3)*2))
    y_2 = -abs((np.sin(x_2)*x_2+x_2)) + 2*noise_2

    df_open2 = list(y_1) + list(y_2)

    df_open2 = list(np.array(df_open2) + np.random.uniform(-25,25))

    df_open1, df_open3 = create_ar2_norm_open(num_1, num_3, df_open2, mu, sigma, ar1, ar2)

    df_close1, df_close2, df_close3 = create_close(df_open1, df_open2, df_open3)


    df_low1, df_high1, df_low2, df_high2, df_low3, df_high3 = create_low_high(df_open1, df_close1, df_open2, df_close2, df_open3, df_close3, h)


    df_open = df_open1+df_open2+df_open3
    df_close = df_close1+df_close2+df_close3
    df_high = df_high1 + df_high2 + df_high3
    df_low = df_low1 + df_low2 + df_low3


    return X, df_open, df_high, df_low, df_close, start, end

def cup_handle(n=500, mu=3, sigma=5, h=1, noise_level=2, ar1 = 0.95, ar2 = 0):

    num_1, num_2, num_3 = num(n)

    X, start, end = X_sart_end(num_1, num_2, num_3)

    x_1 = np.linspace(-10,0,round(num_2/3))
    #x_2 = np.linspace(0,20,round((num_2/3)*2))
    noise_1 = np.random.normal(0,0.7*noise_level,round(num_2/3))
    noise_2 = np.random.normal(0,noise_level,round((num_2/3)*2))

    #y_open2 = x**4 - 4*(x**3)+ x**2 + 7*x - 3
    m = np.random.uniform(0.8,5)
    y_1 = 4*m*(x_1) - 3*noise_1

    n = np.random.uniform(0.2,0.8)

    x_2 = np.linspace(np.pi,2*np.pi,round((num_2/3)*2))
    y_2 = 5*15*np.sin(x_2) + 3*noise_2

    df_open2 = list(y_1) + list(y_2)

    df_open2 = list(np.array(df_open2) + np.random.uniform(-25,25))

    df_open1, df_open3 = create_ar2_norm_open(num_1, num_3, df_open2, mu, sigma, ar1, ar2)

    df_close1, df_close2, df_close3 = create_close(df_open1, df_open2, df_open3)

    df_low1, df_high1, df_low2, df_high2, df_low3, df_high3 = create_low_high(df_open1, df_close1, df_open2, df_close2, df_open3, df_close3, h)

    df_open = df_open1+df_open2+df_open3
    df_close = df_close1+df_close2+df_close3
    df_high = df_high1 + df_high2 + df_high3
    df_low = df_low1 + df_low2 + df_low3



    return X, df_open, df_high, df_low, df_close, start, end

def head_shoulders(n=500, mu=3, sigma=5, h=1, noise_level=2, ar1 = 0.95, ar2 = 0):

    num_1, num_2, num_3 = num(n)

    X, start, end = X_sart_end(num_1, num_2, num_3)

    x_1 = np.linspace(-0.5*np.pi,2*np.pi,round(num_2/3))
    x_2 = np.linspace(0,np.pi,round(num_2/3))

    y_1 = 7*np.sin(x_1) + np.random.normal(0,0.7*noise_level,round(num_2/3))
    m = np.random.uniform(11,25)
    y_2 = m*np.sin(x_2) + np.random.normal(0,0.7*noise_level,round(num_2/3))
    y_3 = 7*np.sin(x_1) + np.random.normal(0,0.7*noise_level,round(num_2/3))
    y_3 = list(y_3)
    y_3.reverse()
    y_1[-1]=0
    y_2[-1]=0
    y_1[0]=0
    y_2[0]=0
    y_3[-1]=0
    y_3[0]=0
    df_open2 = list(y_1) + list(y_2) + list(y_3)

    df_open2 = list(np.array(df_open2) + np.random.uniform(-25,25))

    df_open1, df_open3 = create_ar2_norm_open(num_1, num_3, df_open2, mu, sigma, ar1, ar2)

    df_close1, df_close2, df_close3 = create_close(df_open1, df_open2, df_open3)

    df_low1, df_high1, df_low2, df_high2, df_low3, df_high3 = create_low_high(df_open1, df_close1, df_open2, df_close2, df_open3, df_close3, h)


    df_open = df_open1+df_open2+df_open3
    df_close = df_close1+df_close2+df_close3
    df_high = df_high1 + df_high2 + df_high3
    df_low = df_low1 + df_low2 + df_low3


    return X, df_open, df_high, df_low, df_close, start, end

def inv_head_shoulders(n=500, mu=3, sigma=5, h=1, noise_level=2, ar1 = 0.95, ar2 = 0):

    num_1, num_2, num_3 = num(n)

    X, start, end = X_sart_end(num_1, num_2, num_3)

    x_1 = np.linspace(-0.5*np.pi,2*np.pi,round(num_2/3))
    x_2 = np.linspace(0,np.pi,round(num_2/3))

    y_1 = -7*np.sin(x_1) + np.random.normal(0,0.7*noise_level,round(num_2/3))
    m = np.random.uniform(11,25)
    y_2 = -m*np.sin(x_2) + np.random.normal(0,0.7*noise_level,round(num_2/3))
    y_3 = -7*np.sin(x_1) + np.random.normal(0,0.7*noise_level,round(num_2/3))
    y_3 = list(y_3)
    y_3.reverse()
    y_1[-1]=0
    y_2[-1]=0
    y_1[0]=0
    y_2[0]=0
    y_3[-1]=0
    y_3[0]=0
    df_open2 = list(y_1) + list(y_2) + list(y_3)

    df_open2 = list(np.array(df_open2) + np.random.uniform(-25,25))

    df_open1, df_open3 = create_ar2_norm_open(num_1, num_3, df_open2, mu, sigma, ar1, ar2)

    df_close1, df_close2, df_close3 = create_close(df_open1, df_open2, df_open3)

    df_low1, df_high1, df_low2, df_high2, df_low3, df_high3 = create_low_high(df_open1, df_close1, df_open2, df_close2, df_open3, df_close3, h)

    df_open = df_open1+df_open2+df_open3
    df_close = df_close1+df_close2+df_close3
    df_high = df_high1 + df_high2 + df_high3
    df_low = df_low1 + df_low2 + df_low3


    return X, df_open, df_high, df_low, df_close, start, end

def falling_wedge(n=500, mu=3, sigma=5, h=1, noise_level=2, ar1 = 0.95, ar2 = 0):

    num_1, num_2, num_3 = num(n)

    X, start, end = X_sart_end(num_1, num_2, num_3)

    m1 = np.random.uniform(1.6,2.2)
    m2 = np.random.uniform(1.6,2.2)
    m3 = np.random.uniform(1.6,2.2)
    n1 = np.random.uniform(1,1.5)
    n2 = np.random.uniform(1,1.5)

    rep=int(num_2/5)
    x_1 = np.linspace(0,1.5/(m1+n1),rep)
    x_2 = np.linspace(1.5/(m1+n1),2.5/(m2+n1),rep)
    x_3 = np.linspace(2.5/(m2+n1),4/(m2+n2),rep)
    x_4 = np.linspace(4/(m2+n2),5/(m3+n2),rep)
    x_5 = np.linspace(5/(m3+n2),7/(m3+n2),rep)

    y1 = -m1*x_1
    y2 = n1*x_2-1.5
    y3 = -m2*x_3+1
    y4 = n2*x_4-3
    y5 = -m3*x_5+2

    df_open2 = list(y1) + list(y2) + list(y3) + list(y4) + list(y5)
    x = list(x_1) + list(x_2) + list(x_3) + list(x_4) + list(x_5)


    df_open2 = list(50*np.array(df_open2)+np.random.normal(0,noise_level,len(df_open2)) + np.random.uniform(-25,25))

    df_open1, df_open3 = create_ar2_norm_open(num_1, num_3, df_open2, mu, sigma, ar1, ar2)

    df_close1, df_close2, df_close3 = create_close(df_open1, df_open2, df_open3)

    df_low1, df_high1, df_low2, df_high2, df_low3, df_high3 = create_low_high(df_open1, df_close1, df_open2, df_close2, df_open3, df_close3, h)

    df_open = df_open1+df_open2+df_open3
    df_close = df_close1+df_close2+df_close3
    df_high = df_high1 + df_high2 + df_high3
    df_low = df_low1 + df_low2 + df_low3


    return X, df_open, df_high, df_low, df_close, start, end

def rising_wedge(n=500, mu=3, sigma=5, h=1, noise_level=2, ar1 = 0.95, ar2 = 0):

    num_1, num_2, num_3 = num(n)

    X, start, end = X_sart_end(num_1, num_2, num_3)

    m1 = np.random.uniform(1.6,2.2)
    m2 = np.random.uniform(1.6,2.2)
    m3 = np.random.uniform(1.6,2.2)
    n1 = np.random.uniform(1,1.5)
    n2 = np.random.uniform(1,1.5)

    rep=int(num_2/5)
    x_1 = np.linspace(0,1.5/(m1+n1),rep)
    x_2 = np.linspace(1.5/(m1+n1),2.5/(m2+n1),rep)
    x_3 = np.linspace(2.5/(m2+n1),4/(m2+n2),rep)
    x_4 = np.linspace(4/(m2+n2),5/(m3+n2),rep)
    x_5 = np.linspace(5/(m3+n2),7/(m3+n2),rep)

    y1 = m1*x_1
    y2 = -n1*x_2+1.5
    y3 = m2*x_3-1
    y4 = -n2*x_4+3
    y5 = m3*x_5-2

    df_open2 = list(y1) + list(y2) + list(y3) + list(y4) + list(y5)
    x = list(x_1) + list(x_2) + list(x_3) + list(x_4) + list(x_5)


    df_open2 = list(50*np.array(df_open2)+np.random.normal(0,noise_level,len(df_open2)) + np.random.uniform(-25,25))

    df_open1, df_open3 = create_ar2_norm_open(num_1, num_3, df_open2, mu, sigma, ar1, ar2)

    df_close1, df_close2, df_close3 = create_close(df_open1, df_open2, df_open3)

    df_low1, df_high1, df_low2, df_high2, df_low3, df_high3 = create_low_high(df_open1, df_close1, df_open2, df_close2, df_open3, df_close3, h)

    df_open = df_open1+df_open2+df_open3
    df_close = df_close1+df_close2+df_close3
    df_high = df_high1 + df_high2 + df_high3
    df_low = df_low1 + df_low2 + df_low3


    return X, df_open, df_high, df_low, df_close, start, end

def triple_bottom(n=500, mu=3, sigma=5, h=1, noise_level=2, ar1 = 0.95, ar2 = 0):

    num_1, num_2, num_3 = num(n)

    X, start, end = X_sart_end(num_1, num_2, num_3)

    x_1 = np.linspace(-0.5*np.pi,np.pi,round(num_2/5))
    x_2 = np.linspace(np.pi, 2*np.pi,round(num_2/5))
    x_3 = np.linspace(0,np.pi,round(num_2/5))

    m = np.random.uniform(8,15)

    y_1 = -m*np.sin(x_1) + np.random.normal(0,0.7*noise_level,round(num_2/5))
    y_2 = -(m/3)*np.sin(x_2) + np.random.normal(0,0.7*noise_level,round(num_2/5))
    y_3 = -m*np.sin(x_3) + np.random.normal(0,0.7*noise_level,round(num_2/5))
    y_4 = -(m/3)*np.sin(x_2) + np.random.normal(0,0.7*noise_level,round(num_2/5))
    y_5 = -m*np.sin(x_1) + np.random.normal(0,0.7*noise_level,round(num_2/5))
    y_5 = list(y_5)
    y_5.reverse()
    y_1[-1]=0
    y_2[-1]=0
    y_2[0]=0
    y_4[0]=0
    y_4[-1]=0
    y_5[0]=0
    y_3[-1]=0
    y_3[0]=0
    df_open2 = list(y_1) + list(y_2) + list(y_3) + list(y_4) + list(y_5)

    df_open2 = list(np.array(df_open2) + np.random.uniform(-25,25))

    df_open1, df_open3 = create_ar2_norm_open(num_1, num_3, df_open2, mu, sigma, ar1, ar2)

    df_close1, df_close2, df_close3 = create_close(df_open1, df_open2, df_open3)

    df_low1, df_high1, df_low2, df_high2, df_low3, df_high3 = create_low_high(df_open1, df_close1, df_open2, df_close2, df_open3, df_close3, h)


    df_open = df_open1+df_open2+df_open3
    df_close = df_close1+df_close2+df_close3
    df_high = df_high1 + df_high2 + df_high3
    df_low = df_low1 + df_low2 + df_low3


    return X, df_open, df_high, df_low, df_close, start, end

def triple_top(n=500, mu=3, sigma=5, h=1, noise_level=2, ar1 = 0.95, ar2 = 0):

    num_1, num_2, num_3 = num(n)

    X, start, end = X_sart_end(num_1, num_2, num_3)

    x_1 = np.linspace(-0.5*np.pi,np.pi,round(num_2/5))
    x_2 = np.linspace(np.pi, 2*np.pi,round(num_2/5))
    x_3 = np.linspace(0,np.pi,round(num_2/5))

    m = np.random.uniform(8,15)

    y_1 = m*np.sin(x_1) + np.random.normal(0,0.7*noise_level,round(num_2/5))
    y_2 = (m/3)*np.sin(x_2) + np.random.normal(0,0.7*noise_level,round(num_2/5))
    y_3 = m*np.sin(x_3) + np.random.normal(0,0.7*noise_level,round(num_2/5))
    y_4 = (m/3)*np.sin(x_2) + np.random.normal(0,0.7*noise_level,round(num_2/5))
    y_5 = m*np.sin(x_1) + np.random.normal(0,0.7*noise_level,round(num_2/5))
    y_5 = list(y_5)
    y_5.reverse()
    y_1[-1]=0
    y_2[-1]=0
    y_2[0]=0
    y_4[0]=0
    y_4[-1]=0
    y_5[0]=0
    y_3[-1]=0
    y_3[0]=0
    df_open2 = list(y_1) + list(y_2) + list(y_3) + list(y_4) + list(y_5)

    df_open2 = list(np.array(df_open2) + np.random.uniform(-25,25))

    df_open1, df_open3 = create_ar2_norm_open(num_1, num_3, df_open2, mu, sigma, ar1, ar2)
    #df_open1, df_open3 = create_rand_open(num_1, num_3, df_open2)
    df_close1, df_close2, df_close3 = create_close(df_open1, df_open2, df_open3)

    df_low1, df_high1, df_low2, df_high2, df_low3, df_high3 = create_low_high(df_open1, df_close1, df_open2, df_close2, df_open3, df_close3, h)

    df_open = df_open1+df_open2+df_open3
    df_close = df_close1+df_close2+df_close3
    df_high = df_high1 + df_high2 + df_high3
    df_low = df_low1 + df_low2 + df_low3


    return X, df_open, df_high, df_low, df_close, start, end
