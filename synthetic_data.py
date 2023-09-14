from graph_generator import *


def gen_x_y(l=round(0.4*50),pattern=["rising_wedge","falling_wedge","double_top","double_bottom"], noise=True, general=False, model_type='full'):
    X=[]
    y=[]
    pat={"rising_wedge":1,
            "falling_wedge":2,
            "double_top":3,
            "double_bottom":4
        }
    for i in range(l):

        func={"rising_wedge":rising_wedge(n=500, mu=3, sigma=5, h=1, noise_level=2, ar1 = 0.95, ar2 = 0),
            "falling_wedge":falling_wedge(n=500, mu=3, sigma=5, h=1, noise_level=2, ar1 = 0.95, ar2 = 0),
            "double_top":double_top(n=500, mu=3, sigma=5, h=1, noise_level=2, ar1 = 0.95, ar2 = 0),
            "double_bottom":double_bottom(n=500, mu=3, sigma=5, h=1, noise_level=2, ar1 = 0.95, ar2 = 0),
            }

        date, ope, hig, low, close, start, end = func[pattern]
        ope=np.array(ope)
        hig=np.array(hig)
        low = np.array(low)
        close = np.array(close)
        X.append(np.column_stack((ope, hig, low, close)))
        if model_type == 'full':
            y.append((start, end, pat[pattern]))
        elif model_type == 'categorise':
            tmp_pattern = np.array(1)
            y.append(([tmp_pattern]))

    if noise:
        for i in range(l):
            if pattern == "rising_wedge":
                m= np.random.randint(0,10)
                m = str(m)
                func={"0":double_bottom(n=500, mu=3, sigma=5, h=1, noise_level=2, ar1 = 0.95, ar2 = 0),
                "1":double_top(n=500, mu=3, sigma=5, h=1, noise_level=2, ar1 = 0.95, ar2 = 0),
                "2":bearish_flag(n=500, mu=3, sigma=5, h=1, noise_level=2, ar1 = 0.95, ar2=0),
                "3":bullish_flag(n=500, mu=3, sigma=5, h=1, noise_level=2, ar1 = 0.95, ar2=0),
                "4":cup_handle(n=500, mu=3, sigma=5, h=1, noise_level=2, ar1 = 0.95, ar2 = 0),
                "5":head_shoulders(n=500, mu=3, sigma=5, h=1, noise_level=2, ar1 = 0.95, ar2 = 0),
                "6":inv_head_shoulders(n=500, mu=3, sigma=5, h=1, noise_level=2, ar1 = 0.95, ar2 = 0),
                "7":triple_top(n=500, mu=3, sigma=5, h=1, noise_level=2, ar1 = 0.95, ar2 = 0),
                "8":triple_bottom(n=500, mu=3, sigma=5, h=1, noise_level=2, ar1 = 0.95, ar2 = 0),
                "9":norm_params(ar1=0.95, ar2=0, diff=0, n=500, mu=0, sigma=1, h=2)
                }
                date, ope, hig, low, close, start, end = func[m]
                ope=np.array(ope)
                hig=np.array(hig)
                low = np.array(low)
                close = np.array(close)
                X.append(np.column_stack((ope, hig, low, close)))
                if model_type == 'full':
                    y.append((-1, -1, 0))
                elif model_type == 'categorise':
                    tmp_pattern = np.array(0)
                    y.append(([tmp_pattern]))

            if pattern == "falling_wedge":
                m= np.random.randint(0,10)
                m = str(m)
                func={"0":double_bottom(n=500, mu=3, sigma=5, h=1, noise_level=2, ar1 = 0.95, ar2 = 0),
                "1":double_top(n=500, mu=3, sigma=5, h=1, noise_level=2, ar1 = 0.95, ar2 = 0),
                "2":bearish_flag(n=500, mu=3, sigma=5, h=1, noise_level=2, ar1 = 0.95, ar2=0),
                "3":bullish_flag(n=500, mu=3, sigma=5, h=1, noise_level=2, ar1 = 0.95, ar2=0),
                "4":cup_handle(n=500, mu=3, sigma=5, h=1, noise_level=2, ar1 = 0.95, ar2 = 0),
                "5":head_shoulders(n=500, mu=3, sigma=5, h=1, noise_level=2, ar1 = 0.95, ar2 = 0),
                "6":inv_head_shoulders(n=500, mu=3, sigma=5, h=1, noise_level=2, ar1 = 0.95, ar2 = 0),
                "7":triple_top(n=500, mu=3, sigma=5, h=1, noise_level=2, ar1 = 0.95, ar2 = 0),
                "8":triple_bottom(n=500, mu=3, sigma=5, h=1, noise_level=2, ar1 = 0.95, ar2 = 0),
                "9":norm_params(ar1=0.95, ar2=0, diff=0, n=500, mu=0, sigma=1, h=2)
                }
                date, ope, hig, low, close, start, end = func[m]
                ope=np.array(ope)
                hig=np.array(hig)
                low = np.array(low)
                close = np.array(close)
                X.append(np.column_stack((ope, hig, low, close)))
                if model_type == 'full':
                    y.append((-1, -1, 0))
                elif model_type == 'categorise':
                    tmp_pattern = np.array(0)
                    y.append(([tmp_pattern]))

            if pattern == "double_top":
                m= np.random.randint(0,10)
                m = str(m)
                func={"0":bearish_flag(n=500, mu=3, sigma=5, h=1, noise_level=2, ar1 = 0.95, ar2=0),
                "1":bullish_flag(n=500, mu=3, sigma=5, h=1, noise_level=2, ar1 = 0.95, ar2=0),
                "2":bullish_pennant(n=500, mu=3, sigma=5, h=1, noise_level=2, ar1 = 0.95, ar2 = 0),
                "3":bearish_pennant(n=500, mu=3, sigma=5, h=1, noise_level=2, ar1 = 0.95, ar2 = 0),
                "4":descending_triangle(n=500, mu=3, sigma=5, h=1, noise_level=2, ar1 = 0.95, ar2 = 0),
                "5":ascending_triangle(n=500, mu=3, sigma=5, h=1, noise_level=2, ar1 = 0.95, ar2 = 0),
                "6":cup_handle(n=500, mu=3, sigma=5, h=1, noise_level=2, ar1 = 0.95, ar2 = 0),
                "7":falling_wedge(n=500, mu=3, sigma=5, h=1, noise_level=2, ar1 = 0.95, ar2 = 0),
                "8":rising_wedge(n=500, mu=3, sigma=5, h=1, noise_level=2, ar1 = 0.95, ar2 = 0),
                "9":norm_params(ar1=0.95, ar2=0, diff=0, n=500, mu=0, sigma=1, h=2)
                }
                date, ope, hig, low, close, start, end = func[m]
                ope=np.array(ope)
                hig=np.array(hig)
                low = np.array(low)
                close = np.array(close)
                X.append(np.column_stack((ope, hig, low, close)))
                if model_type == 'full':
                    y.append((-1, -1, 0))
                elif model_type == 'categorise':
                    tmp_pattern = np.array(0)
                    y.append(([tmp_pattern]))

            if pattern == "double_bottom":
                m= np.random.randint(0,10)
                m = str(m)
                func={"0":bearish_flag(n=500, mu=3, sigma=5, h=1, noise_level=2, ar1 = 0.95, ar2=0),
                "1":bullish_flag(n=500, mu=3, sigma=5, h=1, noise_level=2, ar1 = 0.95, ar2=0),
                "2":bullish_pennant(n=500, mu=3, sigma=5, h=1, noise_level=2, ar1 = 0.95, ar2 = 0),
                "3":bearish_pennant(n=500, mu=3, sigma=5, h=1, noise_level=2, ar1 = 0.95, ar2 = 0),
                "4":descending_triangle(n=500, mu=3, sigma=5, h=1, noise_level=2, ar1 = 0.95, ar2 = 0),
                "5":ascending_triangle(n=500, mu=3, sigma=5, h=1, noise_level=2, ar1 = 0.95, ar2 = 0),
                "6":cup_handle(n=500, mu=3, sigma=5, h=1, noise_level=2, ar1 = 0.95, ar2 = 0),
                "7":falling_wedge(n=500, mu=3, sigma=5, h=1, noise_level=2, ar1 = 0.95, ar2 = 0),
                "8":rising_wedge(n=500, mu=3, sigma=5, h=1, noise_level=2, ar1 = 0.95, ar2 = 0),
                "9":norm_params(ar1=0.95, ar2=0, diff=0, n=500, mu=0, sigma=1, h=2)
                }
                date, ope, hig, low, close, start, end = func[m]
                ope=np.array(ope)
                hig=np.array(hig)
                low = np.array(low)
                close = np.array(close)
                X.append(np.column_stack((ope, hig, low, close)))
                if model_type == 'full':
                    y.append((-1, -1, 0))
                elif model_type == 'categorise':
                    tmp_pattern = np.array(0)
                    y.append(([tmp_pattern]))

    if general:
        for i in range(l):
            if pattern == "rising_wedge":
                m= np.random.randint(0,3)
                m = str(m)
                func={"0":bullish_flag(n=500, mu=3, sigma=5, h=1, noise_level=2, ar1 = 0.95, ar2=0),
            "1":bullish_pennant(n=500, mu=3, sigma=5, h=1, noise_level=2, ar1 = 0.95, ar2 = 0),
            "2":ascending_triangle(n=500, mu=3, sigma=5, h=1, noise_level=2, ar1 = 0.95, ar2 = 0),
            }
                date, ope, hig, low, close, start, end = func[m]
                ope=np.array(ope)
                hig=np.array(hig)
                low = np.array(low)
                close = np.array(close)
                X.append(np.column_stack((ope, hig, low, close)))

                if model_type == 'full':
                    y.append((start, end, pat[pattern]))
                elif model_type == 'categorise':
                    tmp_pattern = np.array(1)
                    y.append(([tmp_pattern]))


            if pattern == "falling_wedge":
                m= np.random.randint(0,3)
                m = str(m)
                func={"0":bearish_flag(n=500, mu=3, sigma=5, h=1, noise_level=2, ar1 = 0.95, ar2=0),
            "1":bearish_pennant(n=500, mu=3, sigma=5, h=1, noise_level=2, ar1 = 0.95, ar2 = 0),
            "2":descending_triangle(n=500, mu=3, sigma=5, h=1, noise_level=2, ar1 = 0.95, ar2 = 0)
            }
                date, ope, hig, low, close, start, end = func[m]
                ope=np.array(ope)
                hig=np.array(hig)
                low = np.array(low)
                close = np.array(close)
                X.append(np.column_stack((ope, hig, low, close)))
                if model_type == 'full':
                    y.append((start, end, pat[pattern]))
                elif model_type == 'categorise':
                    tmp_pattern = np.array(1)
                    y.append(([tmp_pattern]))

            if pattern == "double_top":
                m= np.random.randint(0,2)
                m = str(m)
                func={
            "0":head_shoulders(n=500, mu=3, sigma=5, h=1, noise_level=2, ar1 = 0.95, ar2 = 0),
            "1":triple_top(n=500, mu=3, sigma=5, h=1, noise_level=2, ar1 = 0.95, ar2 = 0)
            }
                date, ope, hig, low, close, start, end = func[m]
                ope=np.array(ope)
                hig=np.array(hig)
                low = np.array(low)
                close = np.array(close)
                X.append(np.column_stack((ope, hig, low, close)))
                if model_type == 'full':
                    y.append((start, end, pat[pattern]))
                elif model_type == 'categorise':
                    tmp_pattern = np.array(1)
                    y.append(([tmp_pattern]))

            if pattern == "double_bottom":
                m= np.random.randint(0,2)
                m = str(m)
                func={
            "0":inv_head_shoulders(n=500, mu=3, sigma=5, h=1, noise_level=2, ar1 = 0.95, ar2 = 0),
            "1":triple_bottom(n=500, mu=3, sigma=5, h=1, noise_level=2, ar1 = 0.95, ar2 = 0)
            }
                date, ope, hig, low, close, start, end = func[m]
                ope=np.array(ope)
                hig=np.array(hig)
                low = np.array(low)
                close = np.array(close)
                X.append(np.column_stack((ope, hig, low, close)))
                if model_type == 'full':
                    y.append((start, end, pat[pattern]))
                elif model_type == 'categorise':
                    tmp_pattern = np.array(1)
                    y.append(([tmp_pattern]))


    return X, y



def gen_sec_x_y(l=round(0.4*50),pattern=["ascending_triangle","descending_triangle","h&s_top","h&s_bottom"], noise=True, general=False, model_type='full'):
    X=[]
    y=[]
    pat={"ascending_triangle":5,
            "descending_triangle":6,
            "h&s_top":7,
            "h&s_bottom":8
        }
    for i in range(l):

        func={"ascending_triangle":ascending_triangle(n=500, mu=3, sigma=5, h=1, noise_level=2, ar1 = 0.95, ar2 = 0),
            "descending_triangle":descending_triangle(n=500, mu=3, sigma=5, h=1, noise_level=2, ar1 = 0.95, ar2 = 0),
            "h&s_top":head_shoulders(n=500, mu=3, sigma=5, h=1, noise_level=2, ar1 = 0.95, ar2 = 0),
            "h&s_bottom":inv_head_shoulders(n=500, mu=3, sigma=5, h=1, noise_level=2, ar1 = 0.95, ar2 = 0),
            }

        date, ope, hig, low, close, start, end = func[pattern]
        ope=np.array(ope)
        hig=np.array(hig)
        low = np.array(low)
        close = np.array(close)
        X.append(np.column_stack((ope, hig, low, close)))
        if model_type == 'full':
            y.append((start, end, pat[pattern]))
        elif model_type == 'categorise':
            tmp_pattern = np.array(1)
            y.append(([tmp_pattern]))

    return X, y

def get_X_y(noise=True, general=False):
    amount={"rising_wedge":556,
        "falling_wedge":268,
        "double_top": 393,
        "double_bottom": 348}
    X_ris_wedg, y_ris_wedg = gen_x_y(l=round(amount["rising_wedge"]*0.4), pattern="rising_wedge", noise=noise, general=general)
    X_fal_wedg, y_fal_wedg = gen_x_y(l=round(amount["falling_wedge"]*0.4), pattern="falling_wedge", noise=noise, general=general)
    X_d_top, y_d_top = gen_x_y(l=round(amount["double_top"]*0.4), pattern="double_top", noise=noise, general=general)
    X_d_bottom, y_d_bottom = gen_x_y(l=round(amount["double_bottom"]*0.4), pattern="double_bottom", noise=noise, general=general)
    return X_ris_wedg, y_ris_wedg, X_fal_wedg, y_fal_wedg, X_d_top, y_d_top, X_d_bottom, y_d_bottom


def get_sec_X_y(noise=True, general=False):
    amount={"ascending_triangle":86,
            "descending_triangle":98,
            "h&s_top":86,
            "h&s_bottom":43
        }
    X_ris_wedg, y_ris_wedg = gen_sec_x_y(l=round(amount["ascending_triangle"]), pattern="ascending_triangle", noise=noise, general=general)
    X_fal_wedg, y_fal_wedg = gen_sec_x_y(l=round(amount["descending_triangle"]), pattern="descending_triangle", noise=noise, general=general)
    X_d_top, y_d_top = gen_sec_x_y(l=round(amount["h&s_top"]), pattern="h&s_top", noise=noise, general=general)
    X_d_bottom, y_d_bottom = gen_sec_x_y(l=round(amount["h&s_bottom"]), pattern="h&s_bottom", noise=noise, general=general)
    return X_ris_wedg, y_ris_wedg, X_fal_wedg, y_fal_wedg, X_d_top, y_d_top, X_d_bottom, y_d_bottom


def get_rand_noise(l=round(0.4*50)):
    X=[]
    y=[]

    for i in range(l):

        date, ope, hig, low, close, start, end = norm_params(ar1=0.95, ar2=0, diff=0, n=500, mu=0, sigma=1, h=2)
        ope=np.array(ope)
        hig=np.array(hig)
        low = np.array(low)
        close = np.array(close)
        X.append(np.column_stack((ope, hig, low, close)))
        y.append((-1,-1,0))

    return X, y

def get_noise(X, y, l=round(0.4*50), synth=True, real=True):

    if synth & (real == False):
        X=[]
        y=[]

    if real:

        np.array(y)[:,0]
        start = np.array(y)[:,1]+round(len(np.array(y))*0.1)
        end = np.array(y)[:,1]+round(len(np.array(y))*0.3)
        pattern = np.array(y)[:,2] - y[0][2]
        y = list(np.column_stack((start, end, pattern)))
        if synth:
            num = round(len(X)/2)
            X = X[:num]
            y = y[:num]
    if synth:
        if real:
            l=num

        for i in range(l):

            date, ope, hig, low, close, start, end = norm_params(ar1=0.95, ar2=0, diff=0, n=500, mu=0, sigma=1, h=2)
            ope=np.array(ope)
            hig=np.array(hig)
            low = np.array(low)
            close = np.array(close)
            X.append(np.column_stack((ope, hig, low, close)))
            start=np.random.randint(10,400)
            end=np.random.randint(start,500)
            y.append((start,end,0))

    return X,y

def get_hack_noise(X_ris_wedg, y_ris_wedg, X_fal_wedg, y_fal_wedg, X_d_top, y_d_top, X_d_bottom, y_d_bottom, synth=True, real=True, f=True):
    amount={"rising_wedge":556,
        "falling_wedge":268,
        "double_top": 393,
        "double_bottom": 348}

    if f:
        m=0.4
    else:
        m=1

    X_rw_noise, y_rw_noise = get_noise(X_ris_wedg, y_ris_wedg, l=round(amount["rising_wedge"]*m), synth=synth, real=real)
    X_fw_noise, y_fw_noise = get_noise(X_fal_wedg, y_fal_wedg, l=round(amount["falling_wedge"]*m), synth=synth, real=real)
    X_dt_noise, y_dt_noise = get_noise(X_d_top, y_d_top, l=round(amount["double_top"]*m), synth=synth, real=real)
    X_db_noise, y_db_noise = get_noise(X_d_bottom, y_d_bottom, l=round(amount["double_bottom"]*m), synth=synth, real=real)

    return X_rw_noise, y_rw_noise, X_fw_noise, y_fw_noise, X_dt_noise, y_dt_noise, X_db_noise, y_db_noise
