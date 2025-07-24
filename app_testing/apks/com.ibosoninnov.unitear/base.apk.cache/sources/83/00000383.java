package b.d.b;

import android.graphics.Paint;

/* loaded from: classes.dex */
public /* synthetic */ class m0 {

    /* renamed from: a  reason: collision with root package name */
    public static /* synthetic */ int[] f1638a;

    /* renamed from: b  reason: collision with root package name */
    public static /* synthetic */ int[] f1639b;

    /* renamed from: c  reason: collision with root package name */
    public static /* synthetic */ int[] f1640c;

    /* renamed from: d  reason: collision with root package name */
    public static /* synthetic */ int[] f1641d;

    /* renamed from: e  reason: collision with root package name */
    public static /* synthetic */ int[] f1642e;

    /* renamed from: f  reason: collision with root package name */
    public static /* synthetic */ int[] f1643f;

    /* renamed from: g  reason: collision with root package name */
    public static /* synthetic */ int[] f1644g;

    /* renamed from: h  reason: collision with root package name */
    public static /* synthetic */ int[] f1645h;
    public static /* synthetic */ int[] i;
    public static /* synthetic */ int[] j;
    public static /* synthetic */ int[] k;
    public static /* synthetic */ int[] l;
    public static /* synthetic */ int[] m;
    public static /* synthetic */ int[] n;
    public static /* synthetic */ int[] o;
    public static /* synthetic */ int[] p;
    public static /* synthetic */ int[] q;
    public static /* synthetic */ int[] r;
    public static /* synthetic */ int[] s;
    public static /* synthetic */ int[] t;
    public static /* synthetic */ int[] u;
    public static /* synthetic */ int[] v;

    public static synchronized /* synthetic */ int[] a() {
        int[] iArr;
        synchronized (m0.class) {
            if (j == null) {
                j = g(3);
            }
            iArr = j;
        }
        return iArr;
    }

    public static synchronized /* synthetic */ int[] b() {
        int[] iArr;
        synchronized (m0.class) {
            if (m == null) {
                m = g(2);
            }
            iArr = m;
        }
        return iArr;
    }

    public static synchronized /* synthetic */ int[] c() {
        int[] iArr;
        synchronized (m0.class) {
            if (n == null) {
                n = g(3);
            }
            iArr = n;
        }
        return iArr;
    }

    public static int[] com$airbnb$lottie$model$DocumentData$Justification$s$values() {
        return (int[]) a().clone();
    }

    public static int[] com$airbnb$lottie$model$content$PolystarShape$Type$s$values() {
        return (int[]) b().clone();
    }

    public static int[] com$airbnb$lottie$model$content$ShapeStroke$LineCapType$s$values() {
        return (int[]) c().clone();
    }

    public static int[] com$airbnb$lottie$model$content$ShapeStroke$LineJoinType$s$values() {
        return (int[]) d().clone();
    }

    public static int[] com$airbnb$lottie$model$layer$Layer$MatteType$s$values() {
        return (int[]) e().clone();
    }

    public static synchronized /* synthetic */ int[] d() {
        int[] iArr;
        synchronized (m0.class) {
            if (o == null) {
                o = g(3);
            }
            iArr = o;
        }
        return iArr;
    }

    public static synchronized /* synthetic */ int[] e() {
        int[] iArr;
        synchronized (m0.class) {
            if (q == null) {
                q = g(4);
            }
            iArr = q;
        }
        return iArr;
    }

    public static /* synthetic */ int f(int i2) {
        if (i2 != 0) {
            return i2 - 1;
        }
        throw null;
    }

    public static /* synthetic */ int[] g(int i2) {
        int[] iArr = new int[i2];
        int i3 = 0;
        while (i3 < i2) {
            int i4 = i3 + 1;
            iArr[i3] = i4;
            i3 = i4;
        }
        return iArr;
    }

    public static Paint.Cap h(int i2) {
        int f2 = f(i2);
        if (f2 != 0) {
            if (f2 != 1) {
                return Paint.Cap.SQUARE;
            }
            return Paint.Cap.ROUND;
        }
        return Paint.Cap.BUTT;
    }

    public static Paint.Join i(int i2) {
        int f2 = f(i2);
        if (f2 != 0) {
            if (f2 != 1) {
                if (f2 != 2) {
                    return null;
                }
                return Paint.Join.BEVEL;
            }
            return Paint.Join.ROUND;
        }
        return Paint.Join.MITER;
    }

    public static /* synthetic */ boolean j(int i2) {
        if (i2 == 1 || i2 == 2 || i2 == 3) {
            return false;
        }
        if (i2 == 4 || i2 == 5) {
            return true;
        }
        throw null;
    }

    public static /* synthetic */ int k(int i2) {
        if (i2 == 1) {
            return 1;
        }
        if (i2 == 2) {
            return 2;
        }
        throw null;
    }
}