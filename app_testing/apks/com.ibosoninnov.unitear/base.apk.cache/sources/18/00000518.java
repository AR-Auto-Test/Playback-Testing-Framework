package b.q.b;

import android.graphics.Rect;
import android.util.SparseArray;
import android.view.View;
import android.view.ViewGroup;
import androidx.fragment.app.Fragment;
import b.q.b.q;
import b.q.b.y;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

/* compiled from: FragmentTransition.java */
/* loaded from: classes.dex */
public class f0 {

    /* renamed from: a  reason: collision with root package name */
    public static final int[] f2441a = {0, 3, 0, 1, 5, 4, 7, 6, 9, 8, 10};

    /* renamed from: b  reason: collision with root package name */
    public static final k0 f2442b = new g0();

    /* renamed from: c  reason: collision with root package name */
    public static final k0 f2443c;

    /* compiled from: FragmentTransition.java */
    /* loaded from: classes.dex */
    public interface a {
    }

    /* compiled from: FragmentTransition.java */
    /* loaded from: classes.dex */
    public static class b {

        /* renamed from: a  reason: collision with root package name */
        public Fragment f2444a;

        /* renamed from: b  reason: collision with root package name */
        public boolean f2445b;

        /* renamed from: c  reason: collision with root package name */
        public b.q.b.a f2446c;

        /* renamed from: d  reason: collision with root package name */
        public Fragment f2447d;

        /* renamed from: e  reason: collision with root package name */
        public boolean f2448e;

        /* renamed from: f  reason: collision with root package name */
        public b.q.b.a f2449f;
    }

    static {
        k0 k0Var;
        try {
            k0Var = (k0) Class.forName("b.z.d").getDeclaredConstructor(new Class[0]).newInstance(new Object[0]);
        } catch (Exception unused) {
            k0Var = null;
        }
        f2443c = k0Var;
    }

    public static void a(ArrayList<View> arrayList, b.f.a<String, View> aVar, Collection<String> collection) {
        for (int i = aVar.f1775h - 1; i >= 0; i--) {
            View l = aVar.l(i);
            AtomicInteger atomicInteger = b.j.j.q.f2214a;
            if (collection.contains(l.getTransitionName())) {
                arrayList.add(l);
            }
        }
    }

    /* JADX WARN: Code restructure failed: missing block: B:30:0x0039, code lost:
        if (r0.mAdded != false) goto L67;
     */
    /* JADX WARN: Code restructure failed: missing block: B:60:0x0077, code lost:
        r9 = true;
     */
    /* JADX WARN: Code restructure failed: missing block: B:68:0x0089, code lost:
        if (r0.mHidden == false) goto L67;
     */
    /* JADX WARN: Code restructure failed: missing block: B:69:0x008b, code lost:
        r9 = true;
     */
    /* JADX WARN: Removed duplicated region for block: B:105:? A[ADDED_TO_REGION, RETURN, SYNTHETIC] */
    /* JADX WARN: Removed duplicated region for block: B:74:0x009a  */
    /* JADX WARN: Removed duplicated region for block: B:79:0x00ae A[ADDED_TO_REGION] */
    /* JADX WARN: Removed duplicated region for block: B:92:0x00ce A[ADDED_TO_REGION] */
    /* JADX WARN: Removed duplicated region for block: B:96:0x00d6  */
    /* JADX WARN: Removed duplicated region for block: B:99:0x00e7 A[ADDED_TO_REGION] */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public static void b(b.q.b.a aVar, y.a aVar2, SparseArray<b> sparseArray, boolean z, boolean z2) {
        int i;
        boolean z3;
        boolean z4;
        boolean z5;
        boolean z6;
        b bVar;
        q qVar;
        Fragment fragment = aVar2.f2550b;
        if (fragment == null || (i = fragment.mContainerId) == 0) {
            return;
        }
        int i2 = z ? f2441a[aVar2.f2549a] : aVar2.f2549a;
        boolean z7 = false;
        if (i2 != 1) {
            if (i2 != 3) {
                if (i2 == 4) {
                    boolean z8 = !z2 ? false : false;
                    z5 = z8;
                    z6 = false;
                    z4 = true;
                    bVar = sparseArray.get(i);
                    if (z7) {
                    }
                    if (!z2) {
                    }
                    if (z5) {
                    }
                    if (z2) {
                    }
                } else if (i2 != 5) {
                    if (i2 != 6) {
                        if (i2 != 7) {
                            z6 = false;
                            z4 = false;
                            z5 = false;
                            bVar = sparseArray.get(i);
                            if (z7) {
                                if (bVar == null) {
                                    b bVar2 = new b();
                                    sparseArray.put(i, bVar2);
                                    bVar = bVar2;
                                }
                                bVar.f2444a = fragment;
                                bVar.f2445b = z;
                                bVar.f2446c = aVar;
                            }
                            if (!z2 && z6) {
                                if (bVar != null && bVar.f2447d == fragment) {
                                    bVar.f2447d = null;
                                }
                                qVar = aVar.q;
                                if (fragment.mState < 1 && qVar.m >= 1 && !aVar.p) {
                                    qVar.R(fragment);
                                    qVar.U(fragment, 1);
                                }
                            }
                            if (z5 && (bVar == null || bVar.f2447d == null)) {
                                if (bVar == null) {
                                    b bVar3 = new b();
                                    sparseArray.put(i, bVar3);
                                    bVar = bVar3;
                                }
                                bVar.f2447d = fragment;
                                bVar.f2448e = z;
                                bVar.f2449f = aVar;
                            }
                            if (z2 || !z4 || bVar == null || bVar.f2444a != fragment) {
                                return;
                            }
                            bVar.f2444a = null;
                            return;
                        }
                    }
                } else if (z2) {
                    if (fragment.mHiddenChanged) {
                        if (!fragment.mHidden) {
                        }
                    }
                    z3 = false;
                    z4 = false;
                    z5 = false;
                    z7 = z3;
                    z6 = true;
                    bVar = sparseArray.get(i);
                    if (z7) {
                    }
                    if (!z2) {
                        if (bVar != null) {
                            bVar.f2447d = null;
                        }
                        qVar = aVar.q;
                        if (fragment.mState < 1) {
                            qVar.R(fragment);
                            qVar.U(fragment, 1);
                        }
                    }
                    if (z5) {
                        if (bVar == null) {
                        }
                        bVar.f2447d = fragment;
                        bVar.f2448e = z;
                        bVar.f2449f = aVar;
                    }
                    if (z2) {
                        return;
                    }
                    return;
                } else {
                    z3 = fragment.mHidden;
                    z4 = false;
                    z5 = false;
                    z7 = z3;
                    z6 = true;
                    bVar = sparseArray.get(i);
                    if (z7) {
                    }
                    if (!z2) {
                    }
                    if (z5) {
                    }
                    if (z2) {
                    }
                }
            }
            if (!z2) {
            }
            z5 = z8;
            z6 = false;
            z4 = true;
            bVar = sparseArray.get(i);
            if (z7) {
            }
            if (!z2) {
            }
            if (z5) {
            }
            if (z2) {
            }
        }
        if (z2) {
            z3 = fragment.mIsNewlyAdded;
            z4 = false;
            z5 = false;
            z7 = z3;
            z6 = true;
            bVar = sparseArray.get(i);
            if (z7) {
            }
            if (!z2) {
            }
            if (z5) {
            }
            if (z2) {
            }
        } else {
            if (!fragment.mAdded) {
            }
            z3 = false;
            z4 = false;
            z5 = false;
            z7 = z3;
            z6 = true;
            bVar = sparseArray.get(i);
            if (z7) {
            }
            if (!z2) {
            }
            if (z5) {
            }
            if (z2) {
            }
        }
    }

    public static void c(Fragment fragment, Fragment fragment2, boolean z, b.f.a<String, View> aVar, boolean z2) {
        b.j.b.n enterTransitionCallback;
        if (z) {
            enterTransitionCallback = fragment2.getEnterTransitionCallback();
        } else {
            enterTransitionCallback = fragment.getEnterTransitionCallback();
        }
        if (enterTransitionCallback != null) {
            ArrayList arrayList = new ArrayList();
            ArrayList arrayList2 = new ArrayList();
            int i = aVar == null ? 0 : aVar.f1775h;
            for (int i2 = 0; i2 < i; i2++) {
                arrayList2.add(aVar.h(i2));
                arrayList.add(aVar.l(i2));
            }
            if (z2) {
                throw null;
            }
            throw null;
        }
    }

    public static boolean d(k0 k0Var, List<Object> list) {
        int size = list.size();
        for (int i = 0; i < size; i++) {
            if (!k0Var.e(list.get(i))) {
                return false;
            }
        }
        return true;
    }

    public static b.f.a<String, View> e(k0 k0Var, b.f.a<String, String> aVar, Object obj, b bVar) {
        b.j.b.n enterTransitionCallback;
        ArrayList<String> arrayList;
        Fragment fragment = bVar.f2444a;
        View view = fragment.getView();
        if (!aVar.isEmpty() && obj != null && view != null) {
            b.f.a<String, View> aVar2 = new b.f.a<>();
            k0Var.i(aVar2, view);
            b.q.b.a aVar3 = bVar.f2446c;
            if (bVar.f2445b) {
                enterTransitionCallback = fragment.getExitTransitionCallback();
                arrayList = aVar3.n;
            } else {
                enterTransitionCallback = fragment.getEnterTransitionCallback();
                arrayList = aVar3.o;
            }
            if (arrayList != null) {
                b.f.g.k(aVar2, arrayList);
                b.f.g.k(aVar2, aVar.values());
            }
            if (enterTransitionCallback == null) {
                int i = aVar.f1775h;
                while (true) {
                    i--;
                    if (i < 0) {
                        return aVar2;
                    }
                    if (!aVar2.containsKey(aVar.l(i))) {
                        aVar.j(i);
                    }
                }
            } else {
                throw null;
            }
        } else {
            aVar.clear();
            return null;
        }
    }

    public static b.f.a<String, View> f(k0 k0Var, b.f.a<String, String> aVar, Object obj, b bVar) {
        b.j.b.n exitTransitionCallback;
        ArrayList<String> arrayList;
        if (!aVar.isEmpty() && obj != null) {
            Fragment fragment = bVar.f2447d;
            b.f.a<String, View> aVar2 = new b.f.a<>();
            k0Var.i(aVar2, fragment.requireView());
            b.q.b.a aVar3 = bVar.f2449f;
            if (bVar.f2448e) {
                exitTransitionCallback = fragment.getEnterTransitionCallback();
                arrayList = aVar3.o;
            } else {
                exitTransitionCallback = fragment.getExitTransitionCallback();
                arrayList = aVar3.n;
            }
            if (arrayList != null) {
                b.f.g.k(aVar2, arrayList);
            }
            if (exitTransitionCallback == null) {
                b.f.g.k(aVar, aVar2.keySet());
                return aVar2;
            }
            throw null;
        }
        aVar.clear();
        return null;
    }

    public static k0 g(Fragment fragment, Fragment fragment2) {
        ArrayList arrayList = new ArrayList();
        if (fragment != null) {
            Object exitTransition = fragment.getExitTransition();
            if (exitTransition != null) {
                arrayList.add(exitTransition);
            }
            Object returnTransition = fragment.getReturnTransition();
            if (returnTransition != null) {
                arrayList.add(returnTransition);
            }
            Object sharedElementReturnTransition = fragment.getSharedElementReturnTransition();
            if (sharedElementReturnTransition != null) {
                arrayList.add(sharedElementReturnTransition);
            }
        }
        if (fragment2 != null) {
            Object enterTransition = fragment2.getEnterTransition();
            if (enterTransition != null) {
                arrayList.add(enterTransition);
            }
            Object reenterTransition = fragment2.getReenterTransition();
            if (reenterTransition != null) {
                arrayList.add(reenterTransition);
            }
            Object sharedElementEnterTransition = fragment2.getSharedElementEnterTransition();
            if (sharedElementEnterTransition != null) {
                arrayList.add(sharedElementEnterTransition);
            }
        }
        if (arrayList.isEmpty()) {
            return null;
        }
        k0 k0Var = f2442b;
        if (d(k0Var, arrayList)) {
            return k0Var;
        }
        k0 k0Var2 = f2443c;
        if (k0Var2 == null || !d(k0Var2, arrayList)) {
            throw new IllegalArgumentException("Invalid Transition types");
        }
        return k0Var2;
    }

    public static ArrayList<View> h(k0 k0Var, Object obj, Fragment fragment, ArrayList<View> arrayList, View view) {
        if (obj != null) {
            ArrayList<View> arrayList2 = new ArrayList<>();
            View view2 = fragment.getView();
            if (view2 != null) {
                k0Var.f(arrayList2, view2);
            }
            if (arrayList != null) {
                arrayList2.removeAll(arrayList);
            }
            if (arrayList2.isEmpty()) {
                return arrayList2;
            }
            arrayList2.add(view);
            k0Var.b(obj, arrayList2);
            return arrayList2;
        }
        return null;
    }

    public static Object i(k0 k0Var, Fragment fragment, boolean z) {
        Object enterTransition;
        if (fragment == null) {
            return null;
        }
        if (z) {
            enterTransition = fragment.getReenterTransition();
        } else {
            enterTransition = fragment.getEnterTransition();
        }
        return k0Var.g(enterTransition);
    }

    public static Object j(k0 k0Var, Fragment fragment, boolean z) {
        Object exitTransition;
        if (fragment == null) {
            return null;
        }
        if (z) {
            exitTransition = fragment.getReturnTransition();
        } else {
            exitTransition = fragment.getExitTransition();
        }
        return k0Var.g(exitTransition);
    }

    public static View k(b.f.a<String, View> aVar, b bVar, Object obj, boolean z) {
        ArrayList<String> arrayList;
        String str;
        b.q.b.a aVar2 = bVar.f2446c;
        if (obj == null || aVar == null || (arrayList = aVar2.n) == null || arrayList.isEmpty()) {
            return null;
        }
        if (z) {
            str = aVar2.n.get(0);
        } else {
            str = aVar2.o.get(0);
        }
        return aVar.get(str);
    }

    public static Object l(k0 k0Var, Fragment fragment, Fragment fragment2, boolean z) {
        Object sharedElementEnterTransition;
        if (z) {
            sharedElementEnterTransition = fragment2.getSharedElementReturnTransition();
        } else {
            sharedElementEnterTransition = fragment.getSharedElementEnterTransition();
        }
        return k0Var.w(k0Var.g(sharedElementEnterTransition));
    }

    public static Object m(k0 k0Var, Object obj, Object obj2, Object obj3, Fragment fragment, boolean z) {
        boolean z2;
        if (obj == null || obj2 == null || fragment == null) {
            z2 = true;
        } else if (z) {
            z2 = fragment.getAllowReturnTransitionOverlap();
        } else {
            z2 = fragment.getAllowEnterTransitionOverlap();
        }
        if (z2) {
            return k0Var.m(obj2, obj, obj3);
        }
        return k0Var.l(obj2, obj, obj3);
    }

    public static void n(k0 k0Var, Object obj, Object obj2, b.f.a<String, View> aVar, boolean z, b.q.b.a aVar2) {
        String str;
        ArrayList<String> arrayList = aVar2.n;
        if (arrayList == null || arrayList.isEmpty()) {
            return;
        }
        if (z) {
            str = aVar2.o.get(0);
        } else {
            str = aVar2.n.get(0);
        }
        View view = aVar.get(str);
        k0Var.s(obj, view);
        if (obj2 != null) {
            k0Var.s(obj2, view);
        }
    }

    public static void o(ArrayList<View> arrayList, int i) {
        if (arrayList == null) {
            return;
        }
        for (int size = arrayList.size() - 1; size >= 0; size--) {
            arrayList.get(size).setVisibility(i);
        }
    }

    /* JADX WARN: Removed duplicated region for block: B:101:0x0248  */
    /* JADX WARN: Removed duplicated region for block: B:179:0x045e  */
    /* JADX WARN: Removed duplicated region for block: B:189:0x049c A[SYNTHETIC] */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public static void p(q qVar, ArrayList<b.q.b.a> arrayList, ArrayList<Boolean> arrayList2, int i, int i2, boolean z, a aVar) {
        SparseArray sparseArray;
        int i3;
        int i4;
        Fragment fragment;
        Fragment fragment2;
        k0 g2;
        Fragment fragment3;
        Fragment fragment4;
        ArrayList<View> arrayList3;
        Object obj;
        b.f.a aVar2;
        int i5;
        int i6;
        Object obj2;
        ArrayList<View> arrayList4;
        Object obj3;
        Fragment fragment5;
        ArrayList<View> h2;
        Object m;
        Object obj4;
        Fragment fragment6;
        ArrayList<View> arrayList5;
        boolean z2;
        Fragment fragment7;
        Rect rect;
        Fragment fragment8;
        Fragment fragment9;
        k0 g3;
        ArrayList<View> arrayList6;
        Fragment fragment10;
        boolean z3;
        b.f.a aVar3;
        Object obj5;
        Object m2;
        b.f.a aVar4;
        Object l;
        View view;
        Rect rect2;
        ArrayList<String> arrayList7;
        ArrayList<String> arrayList8;
        q qVar2 = qVar;
        ArrayList<b.q.b.a> arrayList9 = arrayList;
        ArrayList<Boolean> arrayList10 = arrayList2;
        int i7 = i2;
        boolean z4 = z;
        if (qVar2.m < 1) {
            return;
        }
        SparseArray sparseArray2 = new SparseArray();
        for (int i8 = i; i8 < i7; i8++) {
            b.q.b.a aVar5 = arrayList9.get(i8);
            if (arrayList10.get(i8).booleanValue()) {
                if (aVar5.q.o.c()) {
                    for (int size = aVar5.f2541a.size() - 1; size >= 0; size--) {
                        b(aVar5, aVar5.f2541a.get(size), sparseArray2, true, z4);
                    }
                }
            } else {
                int size2 = aVar5.f2541a.size();
                for (int i9 = 0; i9 < size2; i9++) {
                    b(aVar5, aVar5.f2541a.get(i9), sparseArray2, false, z4);
                }
            }
        }
        if (sparseArray2.size() != 0) {
            View view2 = new View(qVar2.n.f2490c);
            int size3 = sparseArray2.size();
            int i10 = 0;
            while (i10 < size3) {
                int keyAt = sparseArray2.keyAt(i10);
                b.f.a aVar6 = new b.f.a();
                int i11 = i7 - 1;
                while (i11 >= i) {
                    b.q.b.a aVar7 = arrayList9.get(i11);
                    if (aVar7.l(keyAt)) {
                        boolean booleanValue = arrayList10.get(i11).booleanValue();
                        ArrayList<String> arrayList11 = aVar7.n;
                        if (arrayList11 != null) {
                            int size4 = arrayList11.size();
                            if (booleanValue) {
                                arrayList8 = aVar7.n;
                                arrayList7 = aVar7.o;
                            } else {
                                ArrayList<String> arrayList12 = aVar7.n;
                                ArrayList<String> arrayList13 = aVar7.o;
                                arrayList7 = arrayList12;
                                arrayList8 = arrayList13;
                            }
                            int i12 = 0;
                            while (i12 < size4) {
                                String str = arrayList7.get(i12);
                                String str2 = arrayList8.get(i12);
                                int i13 = size4;
                                String str3 = (String) aVar6.remove(str2);
                                if (str3 != null) {
                                    aVar6.put(str, str3);
                                } else {
                                    aVar6.put(str, str2);
                                }
                                i12++;
                                size4 = i13;
                            }
                        }
                    }
                    i11--;
                    arrayList9 = arrayList;
                    arrayList10 = arrayList2;
                }
                b bVar = (b) sparseArray2.valueAt(i10);
                if (z4) {
                    ViewGroup viewGroup = qVar2.o.c() ? (ViewGroup) qVar2.o.a(keyAt) : null;
                    if (viewGroup == null || (g3 = g((fragment9 = bVar.f2447d), (fragment8 = bVar.f2444a))) == null) {
                        sparseArray = sparseArray2;
                        i3 = i10;
                        i4 = size3;
                    } else {
                        boolean z5 = bVar.f2445b;
                        boolean z6 = bVar.f2448e;
                        ArrayList<View> arrayList14 = new ArrayList<>();
                        ArrayList<View> arrayList15 = new ArrayList<>();
                        sparseArray = sparseArray2;
                        Object i14 = i(g3, fragment8, z5);
                        Object j = j(g3, fragment9, z6);
                        Fragment fragment11 = bVar.f2444a;
                        i3 = i10;
                        Fragment fragment12 = bVar.f2447d;
                        i4 = size3;
                        if (fragment11 != null) {
                            fragment11.requireView().setVisibility(0);
                        }
                        if (fragment11 == null || fragment12 == null) {
                            arrayList6 = arrayList14;
                            fragment10 = fragment8;
                            z3 = z5;
                        } else {
                            boolean z7 = bVar.f2445b;
                            if (aVar6.isEmpty()) {
                                z3 = z5;
                                l = null;
                            } else {
                                l = l(g3, fragment11, fragment12, z7);
                                z3 = z5;
                            }
                            b.f.a<String, View> f2 = f(g3, aVar6, l, bVar);
                            fragment10 = fragment8;
                            b.f.a<String, View> e2 = e(g3, aVar6, l, bVar);
                            if (aVar6.isEmpty()) {
                                if (f2 != null) {
                                    f2.clear();
                                }
                                if (e2 != null) {
                                    e2.clear();
                                }
                                obj5 = null;
                            } else {
                                a(arrayList15, f2, aVar6.keySet());
                                a(arrayList14, e2, aVar6.values());
                                obj5 = l;
                            }
                            if (i14 == null && j == null && obj5 == null) {
                                arrayList6 = arrayList14;
                            } else {
                                aVar3 = aVar6;
                                c(fragment11, fragment12, z7, f2, true);
                                if (obj5 != null) {
                                    arrayList14.add(view2);
                                    g3.u(obj5, view2, arrayList15);
                                    arrayList6 = arrayList14;
                                    n(g3, obj5, j, f2, bVar.f2448e, bVar.f2449f);
                                    Rect rect3 = new Rect();
                                    View k = k(e2, bVar, i14, z7);
                                    if (k != null) {
                                        g3.r(i14, rect3);
                                    }
                                    view = k;
                                    rect2 = rect3;
                                } else {
                                    arrayList6 = arrayList14;
                                    view = null;
                                    rect2 = null;
                                }
                                b.j.j.k.a(viewGroup, new d0(fragment11, fragment12, z7, e2, view, g3, rect2));
                                if (i14 == null || obj5 != null || j != null) {
                                    ArrayList<View> h3 = h(g3, j, fragment9, arrayList15, view2);
                                    Fragment fragment13 = fragment10;
                                    ArrayList<View> arrayList16 = arrayList6;
                                    ArrayList<View> h4 = h(g3, i14, fragment13, arrayList16, view2);
                                    o(h4, 4);
                                    m2 = m(g3, i14, j, obj5, fragment13, z3);
                                    if (fragment9 != null && h3 != null && (h3.size() > 0 || arrayList15.size() > 0)) {
                                        b.j.f.b bVar2 = new b.j.f.b();
                                        q.b bVar3 = (q.b) aVar;
                                        bVar3.b(fragment9, bVar2);
                                        g3.t(fragment9, m2, bVar2, new z(bVar3, fragment9, bVar2));
                                    }
                                    if (m2 != null) {
                                        if (fragment9 != null && j != null && fragment9.mAdded && fragment9.mHidden && fragment9.mHiddenChanged) {
                                            fragment9.setHideReplaced(true);
                                            g3.p(j, fragment9.getView(), h3);
                                            b.j.j.k.a(fragment9.mContainer, new a0(h3));
                                        }
                                        ArrayList arrayList17 = new ArrayList();
                                        int size5 = arrayList16.size();
                                        for (int i15 = 0; i15 < size5; i15++) {
                                            View view3 = arrayList16.get(i15);
                                            AtomicInteger atomicInteger = b.j.j.q.f2214a;
                                            arrayList17.add(view3.getTransitionName());
                                            view3.setTransitionName(null);
                                        }
                                        g3.q(m2, i14, h4, j, h3, obj5, arrayList16);
                                        g3.c(viewGroup, m2);
                                        int size6 = arrayList16.size();
                                        ArrayList arrayList18 = new ArrayList();
                                        int i16 = 0;
                                        while (i16 < size6) {
                                            View view4 = arrayList15.get(i16);
                                            AtomicInteger atomicInteger2 = b.j.j.q.f2214a;
                                            String transitionName = view4.getTransitionName();
                                            arrayList18.add(transitionName);
                                            if (transitionName == null) {
                                                aVar4 = aVar3;
                                            } else {
                                                view4.setTransitionName(null);
                                                aVar4 = aVar3;
                                                String str4 = (String) aVar4.getOrDefault(transitionName, null);
                                                int i17 = 0;
                                                while (true) {
                                                    if (i17 >= size6) {
                                                        break;
                                                    } else if (str4.equals(arrayList17.get(i17))) {
                                                        arrayList16.get(i17).setTransitionName(transitionName);
                                                        break;
                                                    } else {
                                                        i17++;
                                                    }
                                                }
                                            }
                                            i16++;
                                            aVar3 = aVar4;
                                        }
                                        b.j.j.k.a(viewGroup, new h0(g3, size6, arrayList16, arrayList17, arrayList15, arrayList18));
                                        o(h4, 0);
                                        g3.v(obj5, arrayList15, arrayList16);
                                    }
                                }
                            }
                        }
                        aVar3 = aVar6;
                        obj5 = null;
                        if (i14 == null) {
                        }
                        ArrayList<View> h32 = h(g3, j, fragment9, arrayList15, view2);
                        Fragment fragment132 = fragment10;
                        ArrayList<View> arrayList162 = arrayList6;
                        ArrayList<View> h42 = h(g3, i14, fragment132, arrayList162, view2);
                        o(h42, 4);
                        m2 = m(g3, i14, j, obj5, fragment132, z3);
                        if (fragment9 != null) {
                            b.j.f.b bVar22 = new b.j.f.b();
                            q.b bVar32 = (q.b) aVar;
                            bVar32.b(fragment9, bVar22);
                            g3.t(fragment9, m2, bVar22, new z(bVar32, fragment9, bVar22));
                        }
                        if (m2 != null) {
                        }
                    }
                } else {
                    q qVar3 = qVar2;
                    sparseArray = sparseArray2;
                    i3 = i10;
                    i4 = size3;
                    ViewGroup viewGroup2 = qVar3.o.c() ? (ViewGroup) qVar3.o.a(keyAt) : null;
                    if (viewGroup2 != null && (g2 = g((fragment2 = bVar.f2447d), (fragment = bVar.f2444a))) != null) {
                        boolean z8 = bVar.f2445b;
                        boolean z9 = bVar.f2448e;
                        Object i18 = i(g2, fragment, z8);
                        Object j2 = j(g2, fragment2, z9);
                        ArrayList<View> arrayList19 = new ArrayList<>();
                        ArrayList<View> arrayList20 = new ArrayList<>();
                        Fragment fragment14 = bVar.f2444a;
                        Fragment fragment15 = bVar.f2447d;
                        if (fragment14 != null && fragment15 != null) {
                            boolean z10 = bVar.f2445b;
                            Object l2 = aVar6.isEmpty() ? null : l(g2, fragment14, fragment15, z10);
                            b.f.a<String, View> f3 = f(g2, aVar6, l2, bVar);
                            if (aVar6.isEmpty()) {
                                obj4 = null;
                            } else {
                                arrayList19.addAll(f3.values());
                                obj4 = l2;
                            }
                            if (i18 != null || j2 != null || obj4 != null) {
                                fragment3 = fragment;
                                c(fragment14, fragment15, z10, f3, true);
                                if (obj4 != null) {
                                    rect = new Rect();
                                    g2.u(obj4, view2, arrayList19);
                                    fragment6 = fragment14;
                                    arrayList5 = arrayList20;
                                    z2 = z10;
                                    fragment7 = fragment15;
                                    n(g2, obj4, j2, f3, bVar.f2448e, bVar.f2449f);
                                    if (i18 != null) {
                                        g2.r(i18, rect);
                                    }
                                } else {
                                    fragment6 = fragment14;
                                    arrayList5 = arrayList20;
                                    z2 = z10;
                                    fragment7 = fragment15;
                                    rect = null;
                                }
                                ArrayList<View> arrayList21 = arrayList5;
                                fragment4 = fragment2;
                                obj = j2;
                                Object obj6 = obj4;
                                Object obj7 = obj4;
                                aVar2 = aVar6;
                                arrayList3 = arrayList19;
                                i5 = i3;
                                obj2 = null;
                                arrayList4 = arrayList21;
                                i6 = i4;
                                b.j.j.k.a(viewGroup2, new e0(g2, aVar6, obj6, bVar, arrayList21, view2, fragment6, fragment7, z2, arrayList3, i18, rect));
                                obj3 = obj7;
                                if (i18 == null || obj3 != null || obj != null) {
                                    fragment5 = fragment4;
                                    ArrayList<View> arrayList22 = arrayList3;
                                    h2 = h(g2, obj, fragment5, arrayList22, view2);
                                    if (h2 != null && !h2.isEmpty()) {
                                        obj2 = obj;
                                    }
                                    g2.a(i18, view2);
                                    m = m(g2, i18, obj2, obj3, fragment3, bVar.f2445b);
                                    if (fragment5 != null && h2 != null && (h2.size() > 0 || arrayList22.size() > 0)) {
                                        b.j.f.b bVar4 = new b.j.f.b();
                                        q.b bVar5 = (q.b) aVar;
                                        bVar5.b(fragment5, bVar4);
                                        g2.t(fragment5, m, bVar4, new b0(bVar5, fragment5, bVar4));
                                    }
                                    if (m == null) {
                                        ArrayList<View> arrayList23 = new ArrayList<>();
                                        g2.q(m, i18, arrayList23, obj2, h2, obj3, arrayList4);
                                        b.j.j.k.a(viewGroup2, new c0(i18, g2, view2, fragment3, arrayList4, arrayList23, h2, obj2));
                                        ArrayList<View> arrayList24 = arrayList4;
                                        b.j.j.k.a(viewGroup2, new i0(g2, arrayList24, aVar2));
                                        g2.c(viewGroup2, m);
                                        b.j.j.k.a(viewGroup2, new j0(g2, arrayList24, aVar2));
                                    }
                                }
                                i10 = i5 + 1;
                                qVar2 = qVar;
                                arrayList9 = arrayList;
                                arrayList10 = arrayList2;
                                i7 = i2;
                                z4 = z;
                                size3 = i6;
                                sparseArray2 = sparseArray;
                            }
                        }
                        fragment3 = fragment;
                        fragment4 = fragment2;
                        arrayList3 = arrayList19;
                        obj = j2;
                        aVar2 = aVar6;
                        i5 = i3;
                        i6 = i4;
                        obj2 = null;
                        arrayList4 = arrayList20;
                        obj3 = null;
                        if (i18 == null) {
                        }
                        fragment5 = fragment4;
                        ArrayList<View> arrayList222 = arrayList3;
                        h2 = h(g2, obj, fragment5, arrayList222, view2);
                        if (h2 != null) {
                            obj2 = obj;
                        }
                        g2.a(i18, view2);
                        m = m(g2, i18, obj2, obj3, fragment3, bVar.f2445b);
                        if (fragment5 != null) {
                            b.j.f.b bVar42 = new b.j.f.b();
                            q.b bVar52 = (q.b) aVar;
                            bVar52.b(fragment5, bVar42);
                            g2.t(fragment5, m, bVar42, new b0(bVar52, fragment5, bVar42));
                        }
                        if (m == null) {
                        }
                        i10 = i5 + 1;
                        qVar2 = qVar;
                        arrayList9 = arrayList;
                        arrayList10 = arrayList2;
                        i7 = i2;
                        z4 = z;
                        size3 = i6;
                        sparseArray2 = sparseArray;
                    }
                }
                i5 = i3;
                i6 = i4;
                i10 = i5 + 1;
                qVar2 = qVar;
                arrayList9 = arrayList;
                arrayList10 = arrayList2;
                i7 = i2;
                z4 = z;
                size3 = i6;
                sparseArray2 = sparseArray;
            }
        }
    }
}