package b.q.b;

import android.util.Log;
import androidx.fragment.app.Fragment;
import b.q.b.q;
import b.q.b.y;
import java.io.PrintWriter;
import java.lang.reflect.Modifier;
import java.util.ArrayList;

/* compiled from: BackStackRecord.java */
/* loaded from: classes.dex */
public final class a extends y implements q.e {
    public final q q;
    public boolean r;
    public int s;

    /* JADX WARN: Illegal instructions before constructor call */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public a(q qVar) {
        super(r0, r1 != null ? r1.f2490c.getClassLoader() : null);
        m L = qVar.L();
        n<?> nVar = qVar.n;
        this.s = -1;
        this.q = qVar;
    }

    public static boolean n(y.a aVar) {
        Fragment fragment = aVar.f2550b;
        return (fragment == null || !fragment.mAdded || fragment.mView == null || fragment.mDetached || fragment.mHidden || !fragment.isPostponed()) ? false : true;
    }

    @Override // b.q.b.q.e
    public boolean a(ArrayList<a> arrayList, ArrayList<Boolean> arrayList2) {
        if (q.N(2)) {
            Log.v("FragmentManager", "Run: " + this);
        }
        arrayList.add(this);
        arrayList2.add(Boolean.FALSE);
        if (this.f2547g) {
            q qVar = this.q;
            if (qVar.f2499d == null) {
                qVar.f2499d = new ArrayList<>();
            }
            qVar.f2499d.add(this);
            return true;
        }
        return true;
    }

    @Override // b.q.b.y
    public int c() {
        return g(false);
    }

    @Override // b.q.b.y
    public void d(int i, Fragment fragment, String str, int i2) {
        Class<?> cls = fragment.getClass();
        int modifiers = cls.getModifiers();
        if (!cls.isAnonymousClass() && Modifier.isPublic(modifiers) && (!cls.isMemberClass() || Modifier.isStatic(modifiers))) {
            if (str != null) {
                String str2 = fragment.mTag;
                if (str2 != null && !str.equals(str2)) {
                    throw new IllegalStateException("Can't change tag of fragment " + fragment + ": was " + fragment.mTag + " now " + str);
                }
                fragment.mTag = str;
            }
            if (i != 0) {
                if (i != -1) {
                    int i3 = fragment.mFragmentId;
                    if (i3 != 0 && i3 != i) {
                        throw new IllegalStateException("Can't change container ID of fragment " + fragment + ": was " + fragment.mFragmentId + " now " + i);
                    }
                    fragment.mFragmentId = i;
                    fragment.mContainerId = i;
                } else {
                    throw new IllegalArgumentException("Can't add fragment " + fragment + " with tag " + str + " to container view with no id");
                }
            }
            b(new y.a(i2, fragment));
            fragment.mFragmentManager = this.q;
            return;
        }
        StringBuilder x = c.b.a.a.a.x("Fragment ");
        x.append(cls.getCanonicalName());
        x.append(" must be a public static class to be  properly recreated from instance state.");
        throw new IllegalStateException(x.toString());
    }

    public void e(int i) {
        if (this.f2547g) {
            if (q.N(2)) {
                Log.v("FragmentManager", "Bump nesting in " + this + " by " + i);
            }
            int size = this.f2541a.size();
            for (int i2 = 0; i2 < size; i2++) {
                y.a aVar = this.f2541a.get(i2);
                Fragment fragment = aVar.f2550b;
                if (fragment != null) {
                    fragment.mBackStackNesting += i;
                    if (q.N(2)) {
                        StringBuilder x = c.b.a.a.a.x("Bump nesting of ");
                        x.append(aVar.f2550b);
                        x.append(" to ");
                        x.append(aVar.f2550b.mBackStackNesting);
                        Log.v("FragmentManager", x.toString());
                    }
                }
            }
        }
    }

    public int f() {
        return g(true);
    }

    public int g(boolean z) {
        if (!this.r) {
            if (q.N(2)) {
                Log.v("FragmentManager", "Commit: " + this);
                PrintWriter printWriter = new PrintWriter(new b.j.i.b("FragmentManager"));
                i("  ", printWriter, true);
                printWriter.close();
            }
            this.r = true;
            if (this.f2547g) {
                this.s = this.q.i.getAndIncrement();
            } else {
                this.s = -1;
            }
            this.q.A(this, z);
            return this.s;
        }
        throw new IllegalStateException("commit already called");
    }

    public void h() {
        if (!this.f2547g) {
            this.f2548h = false;
            this.q.D(this, false);
            return;
        }
        throw new IllegalStateException("This transaction is already being added to the back stack");
    }

    public void i(String str, PrintWriter printWriter, boolean z) {
        String str2;
        if (z) {
            printWriter.print(str);
            printWriter.print("mName=");
            printWriter.print(this.i);
            printWriter.print(" mIndex=");
            printWriter.print(this.s);
            printWriter.print(" mCommitted=");
            printWriter.println(this.r);
            if (this.f2546f != 0) {
                printWriter.print(str);
                printWriter.print("mTransition=#");
                printWriter.print(Integer.toHexString(this.f2546f));
            }
            if (this.f2542b != 0 || this.f2543c != 0) {
                printWriter.print(str);
                printWriter.print("mEnterAnim=#");
                printWriter.print(Integer.toHexString(this.f2542b));
                printWriter.print(" mExitAnim=#");
                printWriter.println(Integer.toHexString(this.f2543c));
            }
            if (this.f2544d != 0 || this.f2545e != 0) {
                printWriter.print(str);
                printWriter.print("mPopEnterAnim=#");
                printWriter.print(Integer.toHexString(this.f2544d));
                printWriter.print(" mPopExitAnim=#");
                printWriter.println(Integer.toHexString(this.f2545e));
            }
            if (this.j != 0 || this.k != null) {
                printWriter.print(str);
                printWriter.print("mBreadCrumbTitleRes=#");
                printWriter.print(Integer.toHexString(this.j));
                printWriter.print(" mBreadCrumbTitleText=");
                printWriter.println(this.k);
            }
            if (this.l != 0 || this.m != null) {
                printWriter.print(str);
                printWriter.print("mBreadCrumbShortTitleRes=#");
                printWriter.print(Integer.toHexString(this.l));
                printWriter.print(" mBreadCrumbShortTitleText=");
                printWriter.println(this.m);
            }
        }
        if (this.f2541a.isEmpty()) {
            return;
        }
        printWriter.print(str);
        printWriter.println("Operations:");
        int size = this.f2541a.size();
        for (int i = 0; i < size; i++) {
            y.a aVar = this.f2541a.get(i);
            switch (aVar.f2549a) {
                case 0:
                    str2 = "NULL";
                    break;
                case 1:
                    str2 = "ADD";
                    break;
                case 2:
                    str2 = "REPLACE";
                    break;
                case 3:
                    str2 = "REMOVE";
                    break;
                case 4:
                    str2 = "HIDE";
                    break;
                case 5:
                    str2 = "SHOW";
                    break;
                case 6:
                    str2 = "DETACH";
                    break;
                case 7:
                    str2 = "ATTACH";
                    break;
                case 8:
                    str2 = "SET_PRIMARY_NAV";
                    break;
                case 9:
                    str2 = "UNSET_PRIMARY_NAV";
                    break;
                case 10:
                    str2 = "OP_SET_MAX_LIFECYCLE";
                    break;
                default:
                    StringBuilder x = c.b.a.a.a.x("cmd=");
                    x.append(aVar.f2549a);
                    str2 = x.toString();
                    break;
            }
            printWriter.print(str);
            printWriter.print("  Op #");
            printWriter.print(i);
            printWriter.print(": ");
            printWriter.print(str2);
            printWriter.print(" ");
            printWriter.println(aVar.f2550b);
            if (z) {
                if (aVar.f2551c != 0 || aVar.f2552d != 0) {
                    printWriter.print(str);
                    printWriter.print("enterAnim=#");
                    printWriter.print(Integer.toHexString(aVar.f2551c));
                    printWriter.print(" exitAnim=#");
                    printWriter.println(Integer.toHexString(aVar.f2552d));
                }
                if (aVar.f2553e != 0 || aVar.f2554f != 0) {
                    printWriter.print(str);
                    printWriter.print("popEnterAnim=#");
                    printWriter.print(Integer.toHexString(aVar.f2553e));
                    printWriter.print(" popExitAnim=#");
                    printWriter.println(Integer.toHexString(aVar.f2554f));
                }
            }
        }
    }

    public void j() {
        int size = this.f2541a.size();
        for (int i = 0; i < size; i++) {
            y.a aVar = this.f2541a.get(i);
            Fragment fragment = aVar.f2550b;
            if (fragment != null) {
                fragment.setNextTransition(this.f2546f);
            }
            switch (aVar.f2549a) {
                case 1:
                    fragment.setNextAnim(aVar.f2551c);
                    this.q.f0(fragment, false);
                    this.q.b(fragment);
                    break;
                case 2:
                default:
                    StringBuilder x = c.b.a.a.a.x("Unknown cmd: ");
                    x.append(aVar.f2549a);
                    throw new IllegalArgumentException(x.toString());
                case 3:
                    fragment.setNextAnim(aVar.f2552d);
                    this.q.Z(fragment);
                    break;
                case 4:
                    fragment.setNextAnim(aVar.f2552d);
                    this.q.M(fragment);
                    break;
                case 5:
                    fragment.setNextAnim(aVar.f2551c);
                    this.q.f0(fragment, false);
                    this.q.j0(fragment);
                    break;
                case 6:
                    fragment.setNextAnim(aVar.f2552d);
                    this.q.j(fragment);
                    break;
                case 7:
                    fragment.setNextAnim(aVar.f2551c);
                    this.q.f0(fragment, false);
                    this.q.e(fragment);
                    break;
                case 8:
                    this.q.h0(fragment);
                    break;
                case 9:
                    this.q.h0(null);
                    break;
                case 10:
                    this.q.g0(fragment, aVar.f2556h);
                    break;
            }
            if (!this.p && aVar.f2549a != 1 && fragment != null) {
                this.q.S(fragment);
            }
        }
        if (this.p) {
            return;
        }
        q qVar = this.q;
        qVar.T(qVar.m, true);
    }

    public void k(boolean z) {
        for (int size = this.f2541a.size() - 1; size >= 0; size--) {
            y.a aVar = this.f2541a.get(size);
            Fragment fragment = aVar.f2550b;
            if (fragment != null) {
                int i = this.f2546f;
                fragment.setNextTransition(i != 4097 ? i != 4099 ? i != 8194 ? 0 : 4097 : 4099 : 8194);
            }
            switch (aVar.f2549a) {
                case 1:
                    fragment.setNextAnim(aVar.f2554f);
                    this.q.f0(fragment, true);
                    this.q.Z(fragment);
                    break;
                case 2:
                default:
                    StringBuilder x = c.b.a.a.a.x("Unknown cmd: ");
                    x.append(aVar.f2549a);
                    throw new IllegalArgumentException(x.toString());
                case 3:
                    fragment.setNextAnim(aVar.f2553e);
                    this.q.b(fragment);
                    break;
                case 4:
                    fragment.setNextAnim(aVar.f2553e);
                    this.q.j0(fragment);
                    break;
                case 5:
                    fragment.setNextAnim(aVar.f2554f);
                    this.q.f0(fragment, true);
                    this.q.M(fragment);
                    break;
                case 6:
                    fragment.setNextAnim(aVar.f2553e);
                    this.q.e(fragment);
                    break;
                case 7:
                    fragment.setNextAnim(aVar.f2554f);
                    this.q.f0(fragment, true);
                    this.q.j(fragment);
                    break;
                case 8:
                    this.q.h0(null);
                    break;
                case 9:
                    this.q.h0(fragment);
                    break;
                case 10:
                    this.q.g0(fragment, aVar.f2555g);
                    break;
            }
            if (!this.p && aVar.f2549a != 3 && fragment != null) {
                this.q.S(fragment);
            }
        }
        if (this.p || !z) {
            return;
        }
        q qVar = this.q;
        qVar.T(qVar.m, true);
    }

    public boolean l(int i) {
        int size = this.f2541a.size();
        for (int i2 = 0; i2 < size; i2++) {
            Fragment fragment = this.f2541a.get(i2).f2550b;
            int i3 = fragment != null ? fragment.mContainerId : 0;
            if (i3 != 0 && i3 == i) {
                return true;
            }
        }
        return false;
    }

    public boolean m(ArrayList<a> arrayList, int i, int i2) {
        if (i2 == i) {
            return false;
        }
        int size = this.f2541a.size();
        int i3 = -1;
        for (int i4 = 0; i4 < size; i4++) {
            Fragment fragment = this.f2541a.get(i4).f2550b;
            int i5 = fragment != null ? fragment.mContainerId : 0;
            if (i5 != 0 && i5 != i3) {
                for (int i6 = i; i6 < i2; i6++) {
                    a aVar = arrayList.get(i6);
                    int size2 = aVar.f2541a.size();
                    for (int i7 = 0; i7 < size2; i7++) {
                        Fragment fragment2 = aVar.f2541a.get(i7).f2550b;
                        if ((fragment2 != null ? fragment2.mContainerId : 0) == i5) {
                            return true;
                        }
                    }
                }
                i3 = i5;
            }
        }
        return false;
    }

    public y o(Fragment fragment) {
        q qVar = fragment.mFragmentManager;
        if (qVar != null && qVar != this.q) {
            StringBuilder x = c.b.a.a.a.x("Cannot setPrimaryNavigation for Fragment attached to a different FragmentManager. Fragment ");
            x.append(fragment.toString());
            x.append(" is already attached to a FragmentManager.");
            throw new IllegalStateException(x.toString());
        }
        b(new y.a(8, fragment));
        return this;
    }

    public String toString() {
        StringBuilder sb = new StringBuilder(128);
        sb.append("BackStackEntry{");
        sb.append(Integer.toHexString(System.identityHashCode(this)));
        if (this.s >= 0) {
            sb.append(" #");
            sb.append(this.s);
        }
        if (this.i != null) {
            sb.append(" ");
            sb.append(this.i);
        }
        sb.append("}");
        return sb.toString();
    }
}