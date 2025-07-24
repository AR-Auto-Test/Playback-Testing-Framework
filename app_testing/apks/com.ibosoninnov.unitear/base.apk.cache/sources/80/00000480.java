package b.j.j;

import android.util.Log;
import android.view.View;
import android.view.ViewParent;

/* compiled from: NestedScrollingChildHelper.java */
/* loaded from: classes.dex */
public class f {

    /* renamed from: a  reason: collision with root package name */
    public ViewParent f2204a;

    /* renamed from: b  reason: collision with root package name */
    public ViewParent f2205b;

    /* renamed from: c  reason: collision with root package name */
    public final View f2206c;

    /* renamed from: d  reason: collision with root package name */
    public boolean f2207d;

    /* renamed from: e  reason: collision with root package name */
    public int[] f2208e;

    public f(View view) {
        this.f2206c = view;
    }

    public boolean a(float f2, float f3, boolean z) {
        ViewParent f4;
        if (!this.f2207d || (f4 = f(0)) == null) {
            return false;
        }
        return b.j.b.d.G(f4, this.f2206c, f2, f3, z);
    }

    public boolean b(float f2, float f3) {
        ViewParent f4;
        if (!this.f2207d || (f4 = f(0)) == null) {
            return false;
        }
        return b.j.b.d.H(f4, this.f2206c, f2, f3);
    }

    public boolean c(int i, int i2, int[] iArr, int[] iArr2, int i3) {
        ViewParent f2;
        int i4;
        int i5;
        if (!this.f2207d || (f2 = f(i3)) == null) {
            return false;
        }
        if (i == 0 && i2 == 0) {
            if (iArr2 != null) {
                iArr2[0] = 0;
                iArr2[1] = 0;
                return false;
            }
            return false;
        }
        if (iArr2 != null) {
            this.f2206c.getLocationInWindow(iArr2);
            i4 = iArr2[0];
            i5 = iArr2[1];
        } else {
            i4 = 0;
            i5 = 0;
        }
        if (iArr == null) {
            if (this.f2208e == null) {
                this.f2208e = new int[2];
            }
            iArr = this.f2208e;
        }
        iArr[0] = 0;
        iArr[1] = 0;
        b.j.b.d.I(f2, this.f2206c, i, i2, iArr, i3);
        if (iArr2 != null) {
            this.f2206c.getLocationInWindow(iArr2);
            iArr2[0] = iArr2[0] - i4;
            iArr2[1] = iArr2[1] - i5;
        }
        return (iArr[0] == 0 && iArr[1] == 0) ? false : true;
    }

    public boolean d(int i, int i2, int i3, int i4, int[] iArr) {
        return e(i, i2, i3, i4, iArr, 0, null);
    }

    public final boolean e(int i, int i2, int i3, int i4, int[] iArr, int i5, int[] iArr2) {
        ViewParent f2;
        int i6;
        int i7;
        int[] iArr3;
        if (!this.f2207d || (f2 = f(i5)) == null) {
            return false;
        }
        if (i == 0 && i2 == 0 && i3 == 0 && i4 == 0) {
            if (iArr != null) {
                iArr[0] = 0;
                iArr[1] = 0;
            }
            return false;
        }
        if (iArr != null) {
            this.f2206c.getLocationInWindow(iArr);
            i6 = iArr[0];
            i7 = iArr[1];
        } else {
            i6 = 0;
            i7 = 0;
        }
        if (iArr2 == null) {
            if (this.f2208e == null) {
                this.f2208e = new int[2];
            }
            int[] iArr4 = this.f2208e;
            iArr4[0] = 0;
            iArr4[1] = 0;
            iArr3 = iArr4;
        } else {
            iArr3 = iArr2;
        }
        b.j.b.d.J(f2, this.f2206c, i, i2, i3, i4, i5, iArr3);
        if (iArr != null) {
            this.f2206c.getLocationInWindow(iArr);
            iArr[0] = iArr[0] - i6;
            iArr[1] = iArr[1] - i7;
        }
        return true;
    }

    public final ViewParent f(int i) {
        if (i != 0) {
            if (i != 1) {
                return null;
            }
            return this.f2205b;
        }
        return this.f2204a;
    }

    public boolean g(int i) {
        return f(i) != null;
    }

    public boolean h(int i, int i2) {
        boolean onStartNestedScroll;
        if (f(i2) != null) {
            return true;
        }
        if (this.f2207d) {
            View view = this.f2206c;
            for (ViewParent parent = this.f2206c.getParent(); parent != null; parent = parent.getParent()) {
                View view2 = this.f2206c;
                boolean z = parent instanceof g;
                if (z) {
                    onStartNestedScroll = ((g) parent).onStartNestedScroll(view, view2, i, i2);
                } else {
                    if (i2 == 0) {
                        try {
                            onStartNestedScroll = parent.onStartNestedScroll(view, view2, i);
                        } catch (AbstractMethodError e2) {
                            Log.e("ViewParentCompat", "ViewParent " + parent + " does not implement interface method onStartNestedScroll", e2);
                        }
                    }
                    onStartNestedScroll = false;
                }
                if (onStartNestedScroll) {
                    if (i2 == 0) {
                        this.f2204a = parent;
                    } else if (i2 == 1) {
                        this.f2205b = parent;
                    }
                    View view3 = this.f2206c;
                    if (z) {
                        ((g) parent).onNestedScrollAccepted(view, view3, i, i2);
                    } else if (i2 == 0) {
                        try {
                            parent.onNestedScrollAccepted(view, view3, i);
                        } catch (AbstractMethodError e3) {
                            Log.e("ViewParentCompat", "ViewParent " + parent + " does not implement interface method onNestedScrollAccepted", e3);
                        }
                    }
                    return true;
                }
                if (parent instanceof View) {
                    view = parent;
                }
            }
        }
        return false;
    }

    public void i(int i) {
        ViewParent f2 = f(i);
        if (f2 != null) {
            View view = this.f2206c;
            if (f2 instanceof g) {
                ((g) f2).onStopNestedScroll(view, i);
            } else if (i == 0) {
                try {
                    f2.onStopNestedScroll(view);
                } catch (AbstractMethodError e2) {
                    Log.e("ViewParentCompat", "ViewParent " + f2 + " does not implement interface method onStopNestedScroll", e2);
                }
            }
            if (i == 0) {
                this.f2204a = null;
            } else if (i != 1) {
            } else {
                this.f2205b = null;
            }
        }
    }
}