package b.q.b;

import android.animation.Animator;
import android.app.Activity;
import android.content.Context;
import android.content.res.Configuration;
import android.content.res.Resources;
import android.os.Bundle;
import android.os.Looper;
import android.os.Parcelable;
import android.util.Log;
import android.view.Menu;
import android.view.MenuInflater;
import android.view.MenuItem;
import android.view.View;
import android.view.ViewGroup;
import android.view.animation.Animation;
import androidx.activity.OnBackPressedDispatcher;
import androidx.fragment.app.Fragment;
import b.j.f.b;
import b.q.b.f0;
import b.q.b.u;
import b.q.b.y;
import b.t.e;
import com.google.android.material.internal.StaticLayoutBuilderCompat;
import com.ibosoninnov.unitear.R;
import java.io.FileDescriptor;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;

/* compiled from: FragmentManager.java */
/* loaded from: classes.dex */
public abstract class q {
    public ArrayList<g> A;
    public u B;

    /* renamed from: b  reason: collision with root package name */
    public boolean f2497b;

    /* renamed from: d  reason: collision with root package name */
    public ArrayList<b.q.b.a> f2499d;

    /* renamed from: e  reason: collision with root package name */
    public ArrayList<Fragment> f2500e;

    /* renamed from: g  reason: collision with root package name */
    public OnBackPressedDispatcher f2502g;
    public n<?> n;
    public j o;
    public Fragment p;
    public Fragment q;
    public boolean s;
    public boolean t;
    public boolean u;
    public boolean v;
    public boolean w;
    public ArrayList<b.q.b.a> x;
    public ArrayList<Boolean> y;
    public ArrayList<Fragment> z;

    /* renamed from: a  reason: collision with root package name */
    public final ArrayList<e> f2496a = new ArrayList<>();

    /* renamed from: c  reason: collision with root package name */
    public final x f2498c = new x();

    /* renamed from: f  reason: collision with root package name */
    public final o f2501f = new o(this);

    /* renamed from: h  reason: collision with root package name */
    public final b.a.b f2503h = new a(false);
    public final AtomicInteger i = new AtomicInteger();
    public ConcurrentHashMap<Fragment, HashSet<b.j.f.b>> j = new ConcurrentHashMap<>();
    public final f0.a k = new b();
    public final p l = new p(this);
    public int m = -1;
    public m r = new c();
    public Runnable C = new d();

    /* compiled from: FragmentManager.java */
    /* loaded from: classes.dex */
    public class a extends b.a.b {
        public a(boolean z) {
            super(z);
        }

        @Override // b.a.b
        public void a() {
            q qVar = q.this;
            qVar.C(true);
            if (qVar.f2503h.f530a) {
                qVar.X();
            } else {
                qVar.f2502g.b();
            }
        }
    }

    /* compiled from: FragmentManager.java */
    /* loaded from: classes.dex */
    public class b implements f0.a {
        public b() {
        }

        public void a(Fragment fragment, b.j.f.b bVar) {
            boolean z;
            synchronized (bVar) {
                z = bVar.f2117a;
            }
            if (z) {
                return;
            }
            q qVar = q.this;
            HashSet<b.j.f.b> hashSet = qVar.j.get(fragment);
            if (hashSet != null && hashSet.remove(bVar) && hashSet.isEmpty()) {
                qVar.j.remove(fragment);
                if (fragment.mState < 3) {
                    qVar.i(fragment);
                    qVar.U(fragment, fragment.getStateAfterAnimating());
                }
            }
        }

        public void b(Fragment fragment, b.j.f.b bVar) {
            q qVar = q.this;
            if (qVar.j.get(fragment) == null) {
                qVar.j.put(fragment, new HashSet<>());
            }
            qVar.j.get(fragment).add(bVar);
        }
    }

    /* compiled from: FragmentManager.java */
    /* loaded from: classes.dex */
    public class c extends m {
        public c() {
        }

        @Override // b.q.b.m
        public Fragment a(ClassLoader classLoader, String str) {
            n<?> nVar = q.this.n;
            Context context = nVar.f2490c;
            Objects.requireNonNull(nVar);
            return Fragment.instantiate(context, str, null);
        }
    }

    /* compiled from: FragmentManager.java */
    /* loaded from: classes.dex */
    public class d implements Runnable {
        public d() {
        }

        @Override // java.lang.Runnable
        public void run() {
            q.this.C(true);
        }
    }

    /* compiled from: FragmentManager.java */
    /* loaded from: classes.dex */
    public interface e {
        boolean a(ArrayList<b.q.b.a> arrayList, ArrayList<Boolean> arrayList2);
    }

    /* compiled from: FragmentManager.java */
    /* loaded from: classes.dex */
    public class f implements e {

        /* renamed from: a  reason: collision with root package name */
        public final String f2508a;

        /* renamed from: b  reason: collision with root package name */
        public final int f2509b;

        /* renamed from: c  reason: collision with root package name */
        public final int f2510c;

        public f(String str, int i, int i2) {
            this.f2508a = str;
            this.f2509b = i;
            this.f2510c = i2;
        }

        @Override // b.q.b.q.e
        public boolean a(ArrayList<b.q.b.a> arrayList, ArrayList<Boolean> arrayList2) {
            Fragment fragment = q.this.q;
            if (fragment == null || this.f2509b >= 0 || this.f2508a != null || !fragment.getChildFragmentManager().X()) {
                return q.this.Y(arrayList, arrayList2, this.f2508a, this.f2509b, this.f2510c);
            }
            return false;
        }
    }

    /* compiled from: FragmentManager.java */
    /* loaded from: classes.dex */
    public static class g implements Fragment.f {

        /* renamed from: a  reason: collision with root package name */
        public final boolean f2512a;

        /* renamed from: b  reason: collision with root package name */
        public final b.q.b.a f2513b;

        /* renamed from: c  reason: collision with root package name */
        public int f2514c;

        public g(b.q.b.a aVar, boolean z) {
            this.f2512a = z;
            this.f2513b = aVar;
        }

        public void a() {
            boolean z = this.f2514c > 0;
            for (Fragment fragment : this.f2513b.q.f2498c.g()) {
                fragment.setOnStartEnterTransitionListener(null);
                if (z && fragment.isPostponed()) {
                    fragment.startPostponedEnterTransition();
                }
            }
            b.q.b.a aVar = this.f2513b;
            aVar.q.h(aVar, this.f2512a, !z, true);
        }
    }

    public static boolean N(int i) {
        return Log.isLoggable("FragmentManager", i);
    }

    public void A(e eVar, boolean z) {
        if (!z) {
            if (this.n == null) {
                if (this.v) {
                    throw new IllegalStateException("FragmentManager has been destroyed");
                }
                throw new IllegalStateException("FragmentManager has not been attached to a host.");
            } else if (Q()) {
                throw new IllegalStateException("Can not perform this action after onSaveInstanceState");
            }
        }
        synchronized (this.f2496a) {
            if (this.n == null) {
                if (!z) {
                    throw new IllegalStateException("Activity has been destroyed");
                }
                return;
            }
            this.f2496a.add(eVar);
            e0();
        }
    }

    public final void B(boolean z) {
        if (!this.f2497b) {
            if (this.n == null) {
                if (this.v) {
                    throw new IllegalStateException("FragmentManager has been destroyed");
                }
                throw new IllegalStateException("FragmentManager has not been attached to a host.");
            } else if (Looper.myLooper() == this.n.f2491d.getLooper()) {
                if (!z && Q()) {
                    throw new IllegalStateException("Can not perform this action after onSaveInstanceState");
                }
                if (this.x == null) {
                    this.x = new ArrayList<>();
                    this.y = new ArrayList<>();
                }
                this.f2497b = true;
                try {
                    F(null, null);
                    return;
                } finally {
                    this.f2497b = false;
                }
            } else {
                throw new IllegalStateException("Must be called from main thread of fragment host");
            }
        }
        throw new IllegalStateException("FragmentManager is already executing transactions");
    }

    public boolean C(boolean z) {
        boolean z2;
        B(z);
        boolean z3 = false;
        while (true) {
            ArrayList<b.q.b.a> arrayList = this.x;
            ArrayList<Boolean> arrayList2 = this.y;
            synchronized (this.f2496a) {
                if (this.f2496a.isEmpty()) {
                    z2 = false;
                } else {
                    int size = this.f2496a.size();
                    z2 = false;
                    for (int i = 0; i < size; i++) {
                        z2 |= this.f2496a.get(i).a(arrayList, arrayList2);
                    }
                    this.f2496a.clear();
                    this.n.f2491d.removeCallbacks(this.C);
                }
            }
            if (z2) {
                this.f2497b = true;
                try {
                    a0(this.x, this.y);
                    g();
                    z3 = true;
                } catch (Throwable th) {
                    g();
                    throw th;
                }
            } else {
                l0();
                x();
                this.f2498c.b();
                return z3;
            }
        }
    }

    public void D(e eVar, boolean z) {
        if (z && (this.n == null || this.v)) {
            return;
        }
        B(z);
        ((b.q.b.a) eVar).a(this.x, this.y);
        this.f2497b = true;
        try {
            a0(this.x, this.y);
            g();
            l0();
            x();
            this.f2498c.b();
        } catch (Throwable th) {
            g();
            throw th;
        }
    }

    public final void E(ArrayList<b.q.b.a> arrayList, ArrayList<Boolean> arrayList2, int i, int i2) {
        int i3;
        int i4;
        boolean z;
        int i5;
        int i6;
        ArrayList<Boolean> arrayList3 = arrayList2;
        boolean z2 = arrayList.get(i).p;
        ArrayList<Fragment> arrayList4 = this.z;
        if (arrayList4 == null) {
            this.z = new ArrayList<>();
        } else {
            arrayList4.clear();
        }
        this.z.addAll(this.f2498c.g());
        Fragment fragment = this.q;
        int i7 = i;
        boolean z3 = false;
        while (true) {
            int i8 = 1;
            if (i7 < i2) {
                b.q.b.a aVar = arrayList.get(i7);
                int i9 = 3;
                if (!arrayList3.get(i7).booleanValue()) {
                    ArrayList<Fragment> arrayList5 = this.z;
                    int i10 = 0;
                    while (i10 < aVar.f2541a.size()) {
                        y.a aVar2 = aVar.f2541a.get(i10);
                        int i11 = aVar2.f2549a;
                        if (i11 != i8) {
                            if (i11 == 2) {
                                Fragment fragment2 = aVar2.f2550b;
                                int i12 = fragment2.mContainerId;
                                int size = arrayList5.size() - 1;
                                boolean z4 = false;
                                while (size >= 0) {
                                    Fragment fragment3 = arrayList5.get(size);
                                    if (fragment3.mContainerId != i12) {
                                        i6 = i12;
                                    } else if (fragment3 == fragment2) {
                                        i6 = i12;
                                        z4 = true;
                                    } else {
                                        if (fragment3 == fragment) {
                                            i6 = i12;
                                            aVar.f2541a.add(i10, new y.a(9, fragment3));
                                            i10++;
                                            fragment = null;
                                        } else {
                                            i6 = i12;
                                        }
                                        y.a aVar3 = new y.a(3, fragment3);
                                        aVar3.f2551c = aVar2.f2551c;
                                        aVar3.f2553e = aVar2.f2553e;
                                        aVar3.f2552d = aVar2.f2552d;
                                        aVar3.f2554f = aVar2.f2554f;
                                        aVar.f2541a.add(i10, aVar3);
                                        arrayList5.remove(fragment3);
                                        i10++;
                                    }
                                    size--;
                                    i12 = i6;
                                }
                                if (z4) {
                                    aVar.f2541a.remove(i10);
                                    i10--;
                                } else {
                                    i5 = 1;
                                    aVar2.f2549a = 1;
                                    arrayList5.add(fragment2);
                                    i10 += i5;
                                    i8 = i5;
                                    i9 = 3;
                                }
                            } else if (i11 == i9 || i11 == 6) {
                                arrayList5.remove(aVar2.f2550b);
                                Fragment fragment4 = aVar2.f2550b;
                                if (fragment4 == fragment) {
                                    aVar.f2541a.add(i10, new y.a(9, fragment4));
                                    i10++;
                                    fragment = null;
                                }
                            } else if (i11 == 7) {
                                i5 = 1;
                            } else if (i11 == 8) {
                                aVar.f2541a.add(i10, new y.a(9, fragment));
                                i10++;
                                fragment = aVar2.f2550b;
                            }
                            i5 = 1;
                            i10 += i5;
                            i8 = i5;
                            i9 = 3;
                        } else {
                            i5 = i8;
                        }
                        arrayList5.add(aVar2.f2550b);
                        i10 += i5;
                        i8 = i5;
                        i9 = 3;
                    }
                } else {
                    int i13 = 1;
                    ArrayList<Fragment> arrayList6 = this.z;
                    int size2 = aVar.f2541a.size() - 1;
                    while (size2 >= 0) {
                        y.a aVar4 = aVar.f2541a.get(size2);
                        int i14 = aVar4.f2549a;
                        if (i14 != i13) {
                            if (i14 != 3) {
                                switch (i14) {
                                    case 8:
                                        fragment = null;
                                        break;
                                    case 9:
                                        fragment = aVar4.f2550b;
                                        break;
                                    case 10:
                                        aVar4.f2556h = aVar4.f2555g;
                                        break;
                                }
                                size2--;
                                i13 = 1;
                            }
                            arrayList6.add(aVar4.f2550b);
                            size2--;
                            i13 = 1;
                        }
                        arrayList6.remove(aVar4.f2550b);
                        size2--;
                        i13 = 1;
                    }
                }
                z3 = z3 || aVar.f2547g;
                i7++;
                arrayList3 = arrayList2;
            } else {
                this.z.clear();
                if (!z2) {
                    f0.p(this, arrayList, arrayList2, i, i2, false, this.k);
                }
                int i15 = i;
                while (i15 < i2) {
                    b.q.b.a aVar5 = arrayList.get(i15);
                    if (arrayList2.get(i15).booleanValue()) {
                        aVar5.e(-1);
                        aVar5.k(i15 == i2 + (-1));
                    } else {
                        aVar5.e(1);
                        aVar5.j();
                    }
                    i15++;
                }
                if (z2) {
                    b.f.c<Fragment> cVar = new b.f.c<>(0);
                    a(cVar);
                    i3 = i;
                    int i16 = i2;
                    for (int i17 = i2 - 1; i17 >= i3; i17--) {
                        b.q.b.a aVar6 = arrayList.get(i17);
                        boolean booleanValue = arrayList2.get(i17).booleanValue();
                        int i18 = 0;
                        while (true) {
                            if (i18 >= aVar6.f2541a.size()) {
                                z = false;
                            } else if (b.q.b.a.n(aVar6.f2541a.get(i18))) {
                                z = true;
                            } else {
                                i18++;
                            }
                        }
                        if (z && !aVar6.m(arrayList, i17 + 1, i2)) {
                            if (this.A == null) {
                                this.A = new ArrayList<>();
                            }
                            g gVar = new g(aVar6, booleanValue);
                            this.A.add(gVar);
                            for (int i19 = 0; i19 < aVar6.f2541a.size(); i19++) {
                                y.a aVar7 = aVar6.f2541a.get(i19);
                                if (b.q.b.a.n(aVar7)) {
                                    aVar7.f2550b.setOnStartEnterTransitionListener(gVar);
                                }
                            }
                            if (booleanValue) {
                                aVar6.j();
                            } else {
                                aVar6.k(false);
                            }
                            i16--;
                            if (i17 != i16) {
                                arrayList.remove(i17);
                                arrayList.add(i16, aVar6);
                            }
                            a(cVar);
                        }
                    }
                    int i20 = cVar.j;
                    for (int i21 = 0; i21 < i20; i21++) {
                        Fragment fragment5 = (Fragment) cVar.i[i21];
                        if (!fragment5.mAdded) {
                            View requireView = fragment5.requireView();
                            fragment5.mPostponedAlpha = requireView.getAlpha();
                            requireView.setAlpha(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD);
                        }
                    }
                    i4 = i16;
                } else {
                    i3 = i;
                    i4 = i2;
                }
                if (i4 != i3 && z2) {
                    f0.p(this, arrayList, arrayList2, i, i4, true, this.k);
                    T(this.m, true);
                }
                while (i3 < i2) {
                    b.q.b.a aVar8 = arrayList.get(i3);
                    if (arrayList2.get(i3).booleanValue() && aVar8.s >= 0) {
                        aVar8.s = -1;
                    }
                    Objects.requireNonNull(aVar8);
                    i3++;
                }
                return;
            }
        }
    }

    public final void F(ArrayList<b.q.b.a> arrayList, ArrayList<Boolean> arrayList2) {
        int indexOf;
        int indexOf2;
        ArrayList<g> arrayList3 = this.A;
        int size = arrayList3 == null ? 0 : arrayList3.size();
        int i = 0;
        while (i < size) {
            g gVar = this.A.get(i);
            if (arrayList != null && !gVar.f2512a && (indexOf2 = arrayList.indexOf(gVar.f2513b)) != -1 && arrayList2 != null && arrayList2.get(indexOf2).booleanValue()) {
                this.A.remove(i);
                i--;
                size--;
                b.q.b.a aVar = gVar.f2513b;
                aVar.q.h(aVar, gVar.f2512a, false, false);
            } else {
                if ((gVar.f2514c == 0) || (arrayList != null && gVar.f2513b.m(arrayList, 0, arrayList.size()))) {
                    this.A.remove(i);
                    i--;
                    size--;
                    if (arrayList != null && !gVar.f2512a && (indexOf = arrayList.indexOf(gVar.f2513b)) != -1 && arrayList2 != null && arrayList2.get(indexOf).booleanValue()) {
                        b.q.b.a aVar2 = gVar.f2513b;
                        aVar2.q.h(aVar2, gVar.f2512a, false, false);
                    } else {
                        gVar.a();
                    }
                }
            }
            i++;
        }
    }

    public Fragment G(String str) {
        return this.f2498c.e(str);
    }

    public Fragment H(int i) {
        x xVar = this.f2498c;
        int size = xVar.f2539a.size();
        while (true) {
            size--;
            if (size >= 0) {
                Fragment fragment = xVar.f2539a.get(size);
                if (fragment != null && fragment.mFragmentId == i) {
                    return fragment;
                }
            } else {
                for (w wVar : xVar.f2540b.values()) {
                    if (wVar != null) {
                        Fragment fragment2 = wVar.f2537b;
                        if (fragment2.mFragmentId == i) {
                            return fragment2;
                        }
                    }
                }
                return null;
            }
        }
    }

    public Fragment I(String str) {
        x xVar = this.f2498c;
        Objects.requireNonNull(xVar);
        if (str != null) {
            int size = xVar.f2539a.size();
            while (true) {
                size--;
                if (size < 0) {
                    break;
                }
                Fragment fragment = xVar.f2539a.get(size);
                if (fragment != null && str.equals(fragment.mTag)) {
                    return fragment;
                }
            }
        }
        if (str != null) {
            for (w wVar : xVar.f2540b.values()) {
                if (wVar != null) {
                    Fragment fragment2 = wVar.f2537b;
                    if (str.equals(fragment2.mTag)) {
                        return fragment2;
                    }
                }
            }
        }
        return null;
    }

    public Fragment J(String str) {
        Fragment findFragmentByWho;
        for (w wVar : this.f2498c.f2540b.values()) {
            if (wVar != null && (findFragmentByWho = wVar.f2537b.findFragmentByWho(str)) != null) {
                return findFragmentByWho;
            }
        }
        return null;
    }

    public final ViewGroup K(Fragment fragment) {
        if (fragment.mContainerId > 0 && this.o.c()) {
            View a2 = this.o.a(fragment.mContainerId);
            if (a2 instanceof ViewGroup) {
                return (ViewGroup) a2;
            }
        }
        return null;
    }

    public m L() {
        Fragment fragment = this.p;
        if (fragment != null) {
            return fragment.mFragmentManager.L();
        }
        return this.r;
    }

    public void M(Fragment fragment) {
        if (N(2)) {
            Log.v("FragmentManager", "hide: " + fragment);
        }
        if (fragment.mHidden) {
            return;
        }
        fragment.mHidden = true;
        fragment.mHiddenChanged = true ^ fragment.mHiddenChanged;
        i0(fragment);
    }

    public final boolean O(Fragment fragment) {
        boolean z;
        if (fragment.mHasMenu && fragment.mMenuVisible) {
            return true;
        }
        q qVar = fragment.mChildFragmentManager;
        Iterator it = ((ArrayList) qVar.f2498c.f()).iterator();
        boolean z2 = false;
        while (true) {
            if (!it.hasNext()) {
                z = false;
                break;
            }
            Fragment fragment2 = (Fragment) it.next();
            if (fragment2 != null) {
                z2 = qVar.O(fragment2);
                continue;
            }
            if (z2) {
                z = true;
                break;
            }
        }
        return z;
    }

    public boolean P(Fragment fragment) {
        if (fragment == null) {
            return true;
        }
        q qVar = fragment.mFragmentManager;
        return fragment.equals(qVar.q) && P(qVar.p);
    }

    public boolean Q() {
        return this.t || this.u;
    }

    public void R(Fragment fragment) {
        if (this.f2498c.c(fragment.mWho)) {
            return;
        }
        w wVar = new w(this.l, fragment);
        wVar.a(this.n.f2490c.getClassLoader());
        this.f2498c.f2540b.put(fragment.mWho, wVar);
        if (fragment.mRetainInstanceChangedWhileDetached) {
            if (fragment.mRetainInstance) {
                c(fragment);
            } else {
                b0(fragment);
            }
            fragment.mRetainInstanceChangedWhileDetached = false;
        }
        wVar.f2538c = this.m;
        if (N(2)) {
            Log.v("FragmentManager", "Added fragment to active set " + fragment);
        }
    }

    public void S(Fragment fragment) {
        Animator animator;
        if (!this.f2498c.c(fragment.mWho)) {
            if (N(3)) {
                Log.d("FragmentManager", "Ignoring moving " + fragment + " to state " + this.m + "since it is not added to " + this);
                return;
            }
            return;
        }
        U(fragment, this.m);
        if (fragment.mView != null) {
            x xVar = this.f2498c;
            Objects.requireNonNull(xVar);
            ViewGroup viewGroup = fragment.mContainer;
            View view = fragment.mView;
            Fragment fragment2 = null;
            if (viewGroup != null && view != null) {
                int indexOf = xVar.f2539a.indexOf(fragment);
                while (true) {
                    indexOf--;
                    if (indexOf < 0) {
                        break;
                    }
                    Fragment fragment3 = xVar.f2539a.get(indexOf);
                    if (fragment3.mContainer == viewGroup && fragment3.mView != null) {
                        fragment2 = fragment3;
                        break;
                    }
                }
            }
            if (fragment2 != null) {
                View view2 = fragment2.mView;
                ViewGroup viewGroup2 = fragment.mContainer;
                int indexOfChild = viewGroup2.indexOfChild(view2);
                int indexOfChild2 = viewGroup2.indexOfChild(fragment.mView);
                if (indexOfChild2 < indexOfChild) {
                    viewGroup2.removeViewAt(indexOfChild2);
                    viewGroup2.addView(fragment.mView, indexOfChild);
                }
            }
            if (fragment.mIsNewlyAdded && fragment.mContainer != null) {
                float f2 = fragment.mPostponedAlpha;
                if (f2 > StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD) {
                    fragment.mView.setAlpha(f2);
                }
                fragment.mPostponedAlpha = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
                fragment.mIsNewlyAdded = false;
                h B = b.j.b.d.B(this.n.f2490c, this.o, fragment, true);
                if (B != null) {
                    Animation animation = B.f2467a;
                    if (animation != null) {
                        fragment.mView.startAnimation(animation);
                    } else {
                        B.f2468b.setTarget(fragment.mView);
                        B.f2468b.start();
                    }
                }
            }
        }
        if (fragment.mHiddenChanged) {
            if (fragment.mView != null) {
                h B2 = b.j.b.d.B(this.n.f2490c, this.o, fragment, !fragment.mHidden);
                if (B2 != null && (animator = B2.f2468b) != null) {
                    animator.setTarget(fragment.mView);
                    if (fragment.mHidden) {
                        if (fragment.isHideReplaced()) {
                            fragment.setHideReplaced(false);
                        } else {
                            ViewGroup viewGroup3 = fragment.mContainer;
                            View view3 = fragment.mView;
                            viewGroup3.startViewTransition(view3);
                            B2.f2468b.addListener(new r(this, viewGroup3, view3, fragment));
                        }
                    } else {
                        fragment.mView.setVisibility(0);
                    }
                    B2.f2468b.start();
                } else {
                    if (B2 != null) {
                        fragment.mView.startAnimation(B2.f2467a);
                        B2.f2467a.start();
                    }
                    fragment.mView.setVisibility((!fragment.mHidden || fragment.isHideReplaced()) ? 0 : 8);
                    if (fragment.isHideReplaced()) {
                        fragment.setHideReplaced(false);
                    }
                }
            }
            if (fragment.mAdded && O(fragment)) {
                this.s = true;
            }
            fragment.mHiddenChanged = false;
            fragment.onHiddenChanged(fragment.mHidden);
        }
    }

    public void T(int i, boolean z) {
        n<?> nVar;
        if (this.n == null && i != -1) {
            throw new IllegalStateException("No activity");
        }
        if (z || i != this.m) {
            this.m = i;
            for (Fragment fragment : this.f2498c.g()) {
                S(fragment);
            }
            Iterator it = ((ArrayList) this.f2498c.f()).iterator();
            while (it.hasNext()) {
                Fragment fragment2 = (Fragment) it.next();
                if (fragment2 != null && !fragment2.mIsNewlyAdded) {
                    S(fragment2);
                }
            }
            k0();
            if (this.s && (nVar = this.n) != null && this.m == 4) {
                nVar.l();
                this.s = false;
            }
        }
    }

    /* JADX DEBUG: Failed to insert an additional move for type inference into block B:181:0x03dc */
    /* JADX DEBUG: Failed to insert an additional move for type inference into block B:221:0x04be */
    /* JADX DEBUG: Multi-variable search result rejected for r4v5, resolved type: java.util.HashMap<java.lang.String, b.q.b.w> */
    /* JADX WARN: Code restructure failed: missing block: B:45:0x00ad, code lost:
        if (r1 != 3) goto L48;
     */
    /* JADX WARN: Multi-variable type inference failed */
    /* JADX WARN: Removed duplicated region for block: B:101:0x0211  */
    /* JADX WARN: Removed duplicated region for block: B:153:0x0349  */
    /* JADX WARN: Removed duplicated region for block: B:158:0x0368  */
    /* JADX WARN: Removed duplicated region for block: B:85:0x01b4  */
    /* JADX WARN: Type inference failed for: r12v0 */
    /* JADX WARN: Type inference failed for: r12v1, types: [b.q.b.n<?>, androidx.fragment.app.Fragment, b.q.b.q] */
    /* JADX WARN: Type inference failed for: r12v18 */
    /* JADX WARN: Type inference failed for: r12v19 */
    /* JADX WARN: Type inference failed for: r12v2, types: [java.lang.Object, java.lang.String] */
    /* JADX WARN: Type inference failed for: r12v20 */
    /* JADX WARN: Type inference failed for: r12v21 */
    /* JADX WARN: Type inference failed for: r12v22 */
    /* JADX WARN: Type inference failed for: r12v24 */
    /* JADX WARN: Type inference failed for: r12v3 */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public void U(Fragment fragment, int i) {
        Context context;
        boolean z;
        Fragment e2;
        ViewGroup viewGroup;
        String str;
        w wVar = this.f2498c.f2540b.get(fragment.mWho);
        boolean z2 = true;
        if (wVar == null) {
            wVar = new w(this.l, fragment);
            wVar.f2538c = 1;
        }
        w wVar2 = wVar;
        int i2 = wVar2.f2538c;
        Fragment fragment2 = wVar2.f2537b;
        int i3 = 2;
        if (fragment2.mFromLayout) {
            if (fragment2.mInLayout) {
                i2 = Math.max(i2, 1);
            } else if (i2 < 2) {
                i2 = Math.min(i2, fragment2.mState);
            } else {
                i2 = Math.min(i2, 1);
            }
        }
        if (!wVar2.f2537b.mAdded) {
            i2 = Math.min(i2, 1);
        }
        Fragment fragment3 = wVar2.f2537b;
        if (fragment3.mRemoving) {
            if (fragment3.isInBackStack()) {
                i2 = Math.min(i2, 1);
            } else {
                i2 = Math.min(i2, -1);
            }
        }
        Fragment fragment4 = wVar2.f2537b;
        if (fragment4.mDeferStart && fragment4.mState < 3) {
            i2 = Math.min(i2, 2);
        }
        int ordinal = wVar2.f2537b.mMaxState.ordinal();
        if (ordinal == 2) {
            i2 = Math.min(i2, 1);
        } else if (ordinal == 3) {
            i2 = Math.min(i2, 3);
        } else if (ordinal != 4) {
            i2 = Math.min(i2, -1);
        }
        int min = Math.min(i, i2);
        int i4 = fragment.mState;
        ?? r12 = 0;
        Object obj = null;
        r12 = 0;
        r12 = 0;
        r12 = 0;
        r12 = 0;
        r12 = 0;
        if (i4 <= min) {
            if (i4 < min && !this.j.isEmpty()) {
                f(fragment);
            }
            int i5 = fragment.mState;
            if (i5 != -1) {
                if (i5 != 0) {
                    if (i5 != 1) {
                        if (i5 != 2) {
                        }
                        if (min > i3) {
                            if (N(3)) {
                                c.b.a.a.a.M(c.b.a.a.a.x("moveto STARTED: "), wVar2.f2537b, "FragmentManager");
                            }
                            wVar2.f2537b.performStart();
                            wVar2.f2536a.k(wVar2.f2537b, false);
                        }
                        if (min > 3) {
                            if (N(3)) {
                                c.b.a.a.a.M(c.b.a.a.a.x("moveto RESUMED: "), wVar2.f2537b, "FragmentManager");
                            }
                            wVar2.f2537b.performResume();
                            wVar2.f2536a.i(wVar2.f2537b, false);
                            Fragment fragment5 = wVar2.f2537b;
                            fragment5.mSavedFragmentState = null;
                            fragment5.mSavedViewState = null;
                        }
                    }
                    if (min > -1) {
                        Fragment fragment6 = wVar2.f2537b;
                        if (fragment6.mFromLayout && fragment6.mInLayout && !fragment6.mPerformedCreateView) {
                            if (N(3)) {
                                c.b.a.a.a.M(c.b.a.a.a.x("moveto CREATE_VIEW: "), wVar2.f2537b, "FragmentManager");
                            }
                            Fragment fragment7 = wVar2.f2537b;
                            fragment7.performCreateView(fragment7.performGetLayoutInflater(fragment7.mSavedFragmentState), null, wVar2.f2537b.mSavedFragmentState);
                            View view = wVar2.f2537b.mView;
                            if (view != null) {
                                view.setSaveFromParentEnabled(false);
                                Fragment fragment8 = wVar2.f2537b;
                                fragment8.mView.setTag(R.id.fragment_container_view_tag, fragment8);
                                Fragment fragment9 = wVar2.f2537b;
                                if (fragment9.mHidden) {
                                    fragment9.mView.setVisibility(8);
                                }
                                Fragment fragment10 = wVar2.f2537b;
                                fragment10.onViewCreated(fragment10.mView, fragment10.mSavedFragmentState);
                                p pVar = wVar2.f2536a;
                                Fragment fragment11 = wVar2.f2537b;
                                pVar.m(fragment11, fragment11.mView, fragment11.mSavedFragmentState, false);
                            }
                        }
                    }
                    if (min > 1) {
                        j jVar = this.o;
                        if (!wVar2.f2537b.mFromLayout) {
                            if (N(3)) {
                                c.b.a.a.a.M(c.b.a.a.a.x("moveto CREATE_VIEW: "), wVar2.f2537b, "FragmentManager");
                            }
                            Fragment fragment12 = wVar2.f2537b;
                            ViewGroup viewGroup2 = fragment12.mContainer;
                            if (viewGroup2 == null) {
                                int i6 = fragment12.mContainerId;
                                if (i6 == 0) {
                                    viewGroup2 = null;
                                } else if (i6 != -1) {
                                    viewGroup2 = (ViewGroup) jVar.a(i6);
                                    if (viewGroup2 == null) {
                                        Fragment fragment13 = wVar2.f2537b;
                                        if (!fragment13.mRestored) {
                                            try {
                                                str = fragment13.getResources().getResourceName(wVar2.f2537b.mContainerId);
                                            } catch (Resources.NotFoundException unused) {
                                                str = "unknown";
                                            }
                                            StringBuilder x = c.b.a.a.a.x("No view found for id 0x");
                                            x.append(Integer.toHexString(wVar2.f2537b.mContainerId));
                                            x.append(" (");
                                            x.append(str);
                                            x.append(") for fragment ");
                                            x.append(wVar2.f2537b);
                                            throw new IllegalArgumentException(x.toString());
                                        }
                                    }
                                } else {
                                    StringBuilder x2 = c.b.a.a.a.x("Cannot create fragment ");
                                    x2.append(wVar2.f2537b);
                                    x2.append(" for a container view with no id");
                                    throw new IllegalArgumentException(x2.toString());
                                }
                            }
                            Fragment fragment14 = wVar2.f2537b;
                            fragment14.mContainer = viewGroup2;
                            fragment14.performCreateView(fragment14.performGetLayoutInflater(fragment14.mSavedFragmentState), viewGroup2, wVar2.f2537b.mSavedFragmentState);
                            View view2 = wVar2.f2537b.mView;
                            if (view2 != null) {
                                view2.setSaveFromParentEnabled(false);
                                Fragment fragment15 = wVar2.f2537b;
                                fragment15.mView.setTag(R.id.fragment_container_view_tag, fragment15);
                                if (viewGroup2 != null) {
                                    viewGroup2.addView(wVar2.f2537b.mView);
                                }
                                Fragment fragment16 = wVar2.f2537b;
                                if (fragment16.mHidden) {
                                    fragment16.mView.setVisibility(8);
                                }
                                View view3 = wVar2.f2537b.mView;
                                AtomicInteger atomicInteger = b.j.j.q.f2214a;
                                view3.requestApplyInsets();
                                Fragment fragment17 = wVar2.f2537b;
                                fragment17.onViewCreated(fragment17.mView, fragment17.mSavedFragmentState);
                                p pVar2 = wVar2.f2536a;
                                Fragment fragment18 = wVar2.f2537b;
                                pVar2.m(fragment18, fragment18.mView, fragment18.mSavedFragmentState, false);
                                Fragment fragment19 = wVar2.f2537b;
                                fragment19.mIsNewlyAdded = (fragment19.mView.getVisibility() != 0 || wVar2.f2537b.mContainer == null) ? false : false;
                            }
                        }
                        if (N(3)) {
                            c.b.a.a.a.M(c.b.a.a.a.x("moveto ACTIVITY_CREATED: "), wVar2.f2537b, "FragmentManager");
                        }
                        Fragment fragment20 = wVar2.f2537b;
                        fragment20.performActivityCreated(fragment20.mSavedFragmentState);
                        p pVar3 = wVar2.f2536a;
                        Fragment fragment21 = wVar2.f2537b;
                        pVar3.a(fragment21, fragment21.mSavedFragmentState, false);
                        if (N(3)) {
                            c.b.a.a.a.M(c.b.a.a.a.x("moveto RESTORE_VIEW_STATE: "), wVar2.f2537b, "FragmentManager");
                        }
                        Fragment fragment22 = wVar2.f2537b;
                        if (fragment22.mView != null) {
                            fragment22.restoreViewState(fragment22.mSavedFragmentState);
                        }
                        wVar2.f2537b.mSavedFragmentState = null;
                    }
                    i3 = 2;
                    if (min > i3) {
                    }
                    if (min > 3) {
                    }
                }
            } else if (min > -1) {
                if (N(3)) {
                    Log.d("FragmentManager", "moveto ATTACHED: " + fragment);
                }
                Fragment fragment23 = fragment.mTarget;
                if (fragment23 != null) {
                    if (fragment23.equals(G(fragment23.mWho))) {
                        Fragment fragment24 = fragment.mTarget;
                        if (fragment24.mState < 1) {
                            U(fragment24, 1);
                        }
                        fragment.mTargetWho = fragment.mTarget.mWho;
                        fragment.mTarget = null;
                    } else {
                        throw new IllegalStateException("Fragment " + fragment + " declared target fragment " + fragment.mTarget + " that does not belong to this FragmentManager!");
                    }
                }
                String str2 = fragment.mTargetWho;
                if (str2 != null) {
                    Fragment e3 = this.f2498c.e(str2);
                    if (e3 != null) {
                        if (e3.mState < 1) {
                            U(e3, 1);
                        }
                    } else {
                        StringBuilder sb = new StringBuilder();
                        sb.append("Fragment ");
                        sb.append(fragment);
                        sb.append(" declared target fragment ");
                        throw new IllegalStateException(c.b.a.a.a.v(sb, fragment.mTargetWho, " that does not belong to this FragmentManager!"));
                    }
                }
                n<?> nVar = this.n;
                Fragment fragment25 = this.p;
                Fragment fragment26 = wVar2.f2537b;
                fragment26.mHost = nVar;
                fragment26.mParentFragment = fragment25;
                fragment26.mFragmentManager = this;
                wVar2.f2536a.g(fragment26, nVar.f2490c, false);
                wVar2.f2537b.performAttach();
                Fragment fragment27 = wVar2.f2537b;
                Fragment fragment28 = fragment27.mParentFragment;
                if (fragment28 == null) {
                    nVar.d(fragment27);
                } else {
                    fragment28.onAttachFragment(fragment27);
                }
                wVar2.f2536a.b(wVar2.f2537b, nVar.f2490c, false);
            }
            if (min > 0) {
                if (N(3)) {
                    c.b.a.a.a.M(c.b.a.a.a.x("moveto CREATED: "), wVar2.f2537b, "FragmentManager");
                }
                Fragment fragment29 = wVar2.f2537b;
                if (!fragment29.mIsCreated) {
                    wVar2.f2536a.h(fragment29, fragment29.mSavedFragmentState, false);
                    Fragment fragment30 = wVar2.f2537b;
                    fragment30.performCreate(fragment30.mSavedFragmentState);
                    p pVar4 = wVar2.f2536a;
                    Fragment fragment31 = wVar2.f2537b;
                    pVar4.c(fragment31, fragment31.mSavedFragmentState, false);
                } else {
                    fragment29.restoreChildFragmentState(fragment29.mSavedFragmentState);
                    wVar2.f2537b.mState = 1;
                }
            }
            if (min > -1) {
            }
            if (min > 1) {
            }
            i3 = 2;
            if (min > i3) {
            }
            if (min > 3) {
            }
        } else if (i4 > min) {
            if (i4 != 0) {
                if (i4 != 1) {
                    int i7 = 2;
                    if (i4 != 2) {
                        if (i4 != 3) {
                            if (i4 == 4) {
                                if (min < 4) {
                                    if (N(3)) {
                                        c.b.a.a.a.M(c.b.a.a.a.x("movefrom RESUMED: "), wVar2.f2537b, "FragmentManager");
                                    }
                                    wVar2.f2537b.performPause();
                                    wVar2.f2536a.f(wVar2.f2537b, false);
                                }
                            }
                        }
                        if (min < 3) {
                            if (N(3)) {
                                c.b.a.a.a.M(c.b.a.a.a.x("movefrom STARTED: "), wVar2.f2537b, "FragmentManager");
                            }
                            wVar2.f2537b.performStop();
                            wVar2.f2536a.l(wVar2.f2537b, false);
                        }
                        i7 = 2;
                    }
                    if (min < i7) {
                        if (N(3)) {
                            Log.d("FragmentManager", "movefrom ACTIVITY_CREATED: " + fragment);
                        }
                        if (fragment.mView != null && this.n.h(fragment) && fragment.mSavedViewState == null) {
                            wVar2.b();
                        }
                        View view4 = fragment.mView;
                        if (view4 != null && (viewGroup = fragment.mContainer) != null) {
                            viewGroup.endViewTransition(view4);
                            fragment.mView.clearAnimation();
                            if (!fragment.isRemovingParent()) {
                                h B = (this.m <= -1 || this.v || fragment.mView.getVisibility() != 0 || fragment.mPostponedAlpha < StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD) ? null : b.j.b.d.B(this.n.f2490c, this.o, fragment, false);
                                fragment.mPostponedAlpha = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
                                ViewGroup viewGroup3 = fragment.mContainer;
                                View view5 = fragment.mView;
                                if (B != null) {
                                    fragment.setStateAfterAnimating(min);
                                    f0.a aVar = this.k;
                                    View view6 = fragment.mView;
                                    ViewGroup viewGroup4 = fragment.mContainer;
                                    viewGroup4.startViewTransition(view6);
                                    b.j.f.b bVar = new b.j.f.b();
                                    bVar.a(new b.q.b.e(fragment));
                                    b bVar2 = (b) aVar;
                                    bVar2.b(fragment, bVar);
                                    if (B.f2467a != null) {
                                        i iVar = new i(B.f2467a, viewGroup4, view6);
                                        fragment.setAnimatingAway(fragment.mView);
                                        iVar.setAnimationListener(new b.q.b.f(viewGroup4, fragment, bVar2, bVar));
                                        fragment.mView.startAnimation(iVar);
                                        obj = null;
                                    } else {
                                        Animator animator = B.f2468b;
                                        fragment.setAnimator(animator);
                                        animator.addListener(new b.q.b.g(viewGroup4, view6, fragment, bVar2, bVar));
                                        animator.setTarget(fragment.mView);
                                        animator.start();
                                        obj = null;
                                    }
                                }
                                viewGroup3.removeView(view5);
                                r12 = obj;
                                if (viewGroup3 != fragment.mContainer) {
                                    return;
                                }
                            }
                        }
                        if (this.j.get(fragment) == null) {
                            i(fragment);
                        } else {
                            fragment.setStateAfterAnimating(min);
                        }
                    }
                }
                if (min < 1) {
                    if (!(fragment.mRemoving && !fragment.isInBackStack()) && !this.B.c(fragment)) {
                        String str3 = fragment.mTargetWho;
                        if (str3 != null && (e2 = this.f2498c.e(str3)) != null && e2.getRetainInstance()) {
                            fragment.mTarget = e2;
                        }
                    } else {
                        Fragment fragment32 = wVar2.f2537b;
                        if (this.f2498c.c(fragment32.mWho)) {
                            if (N(2)) {
                                Log.v("FragmentManager", "Removed fragment from active set " + fragment32);
                            }
                            x xVar = this.f2498c;
                            Objects.requireNonNull(xVar);
                            Fragment fragment33 = wVar2.f2537b;
                            for (w wVar3 : xVar.f2540b.values()) {
                                if (wVar3 != null) {
                                    Fragment fragment34 = wVar3.f2537b;
                                    if (fragment33.mWho.equals(fragment34.mTargetWho)) {
                                        fragment34.mTarget = fragment33;
                                        fragment34.mTargetWho = r12;
                                    }
                                }
                            }
                            xVar.f2540b.put(fragment33.mWho, r12);
                            String str4 = fragment33.mTargetWho;
                            if (str4 != null) {
                                fragment33.mTarget = xVar.e(str4);
                            }
                            b0(fragment32);
                        }
                    }
                    if (this.j.get(fragment) != null) {
                        fragment.setStateAfterAnimating(min);
                        min = 1;
                    } else {
                        n<?> nVar2 = this.n;
                        u uVar = this.B;
                        if (N(3)) {
                            c.b.a.a.a.M(c.b.a.a.a.x("movefrom CREATED: "), wVar2.f2537b, "FragmentManager");
                        }
                        Fragment fragment35 = wVar2.f2537b;
                        boolean z3 = fragment35.mRemoving && !fragment35.isInBackStack();
                        if (z3 || uVar.c(wVar2.f2537b)) {
                            if (nVar2 instanceof b.t.z) {
                                z = uVar.f2528h;
                            } else {
                                z = nVar2.f2490c instanceof Activity ? !((Activity) context).isChangingConfigurations() : true;
                            }
                            if (z3 || z) {
                                Fragment fragment36 = wVar2.f2537b;
                                Objects.requireNonNull(uVar);
                                if (N(3)) {
                                    Log.d("FragmentManager", "Clearing non-config state for " + fragment36);
                                }
                                u uVar2 = uVar.f2525e.get(fragment36.mWho);
                                if (uVar2 != null) {
                                    uVar2.a();
                                    uVar.f2525e.remove(fragment36.mWho);
                                }
                                b.t.y yVar = uVar.f2526f.get(fragment36.mWho);
                                if (yVar != null) {
                                    yVar.a();
                                    uVar.f2526f.remove(fragment36.mWho);
                                }
                            }
                            wVar2.f2537b.performDestroy();
                            wVar2.f2536a.d(wVar2.f2537b, false);
                        } else {
                            wVar2.f2537b.mState = 0;
                        }
                    }
                }
            }
            if (min < 0) {
                u uVar3 = this.B;
                if (N(3)) {
                    c.b.a.a.a.M(c.b.a.a.a.x("movefrom ATTACHED: "), wVar2.f2537b, "FragmentManager");
                }
                wVar2.f2537b.performDetach();
                wVar2.f2536a.e(wVar2.f2537b, false);
                Fragment fragment37 = wVar2.f2537b;
                fragment37.mState = -1;
                fragment37.mHost = r12;
                fragment37.mParentFragment = r12;
                fragment37.mFragmentManager = r12;
                if (((!fragment37.mRemoving || fragment37.isInBackStack()) ? false : false) || uVar3.c(wVar2.f2537b)) {
                    if (N(3)) {
                        c.b.a.a.a.M(c.b.a.a.a.x("initState called for fragment: "), wVar2.f2537b, "FragmentManager");
                    }
                    wVar2.f2537b.initState();
                }
            }
        }
        if (fragment.mState != min) {
            if (N(3)) {
                Log.d("FragmentManager", "moveToState: Fragment state for " + fragment + " not updated inline; expected state " + min + " found " + fragment.mState);
            }
            fragment.mState = min;
        }
    }

    public void V() {
        if (this.n == null) {
            return;
        }
        this.t = false;
        this.u = false;
        for (Fragment fragment : this.f2498c.g()) {
            if (fragment != null) {
                fragment.noteStateNotSaved();
            }
        }
    }

    public void W(Fragment fragment) {
        if (fragment.mDeferStart) {
            if (this.f2497b) {
                this.w = true;
                return;
            }
            fragment.mDeferStart = false;
            U(fragment, this.m);
        }
    }

    public boolean X() {
        C(false);
        B(true);
        Fragment fragment = this.q;
        if (fragment == null || !fragment.getChildFragmentManager().X()) {
            boolean Y = Y(this.x, this.y, null, -1, 0);
            if (Y) {
                this.f2497b = true;
                try {
                    a0(this.x, this.y);
                } finally {
                    g();
                }
            }
            l0();
            x();
            this.f2498c.b();
            return Y;
        }
        return true;
    }

    public boolean Y(ArrayList<b.q.b.a> arrayList, ArrayList<Boolean> arrayList2, String str, int i, int i2) {
        ArrayList<b.q.b.a> arrayList3 = this.f2499d;
        if (arrayList3 == null) {
            return false;
        }
        if (str == null && i < 0 && (i2 & 1) == 0) {
            int size = arrayList3.size() - 1;
            if (size < 0) {
                return false;
            }
            arrayList.add(this.f2499d.remove(size));
            arrayList2.add(Boolean.TRUE);
        } else {
            int i3 = -1;
            if (str != null || i >= 0) {
                int size2 = arrayList3.size() - 1;
                while (size2 >= 0) {
                    b.q.b.a aVar = this.f2499d.get(size2);
                    if ((str != null && str.equals(aVar.i)) || (i >= 0 && i == aVar.s)) {
                        break;
                    }
                    size2--;
                }
                if (size2 < 0) {
                    return false;
                }
                if ((i2 & 1) != 0) {
                    while (true) {
                        size2--;
                        if (size2 < 0) {
                            break;
                        }
                        b.q.b.a aVar2 = this.f2499d.get(size2);
                        if (str == null || !str.equals(aVar2.i)) {
                            if (i < 0 || i != aVar2.s) {
                                break;
                            }
                        }
                    }
                }
                i3 = size2;
            }
            if (i3 == this.f2499d.size() - 1) {
                return false;
            }
            for (int size3 = this.f2499d.size() - 1; size3 > i3; size3--) {
                arrayList.add(this.f2499d.remove(size3));
                arrayList2.add(Boolean.TRUE);
            }
        }
        return true;
    }

    public void Z(Fragment fragment) {
        if (N(2)) {
            Log.v("FragmentManager", "remove: " + fragment + " nesting=" + fragment.mBackStackNesting);
        }
        boolean z = !fragment.isInBackStack();
        if (!fragment.mDetached || z) {
            this.f2498c.h(fragment);
            if (O(fragment)) {
                this.s = true;
            }
            fragment.mRemoving = true;
            i0(fragment);
        }
    }

    public final void a(b.f.c<Fragment> cVar) {
        int i = this.m;
        if (i < 1) {
            return;
        }
        int min = Math.min(i, 3);
        for (Fragment fragment : this.f2498c.g()) {
            if (fragment.mState < min) {
                U(fragment, min);
                if (fragment.mView != null && !fragment.mHidden && fragment.mIsNewlyAdded) {
                    cVar.add(fragment);
                }
            }
        }
    }

    public final void a0(ArrayList<b.q.b.a> arrayList, ArrayList<Boolean> arrayList2) {
        if (arrayList.isEmpty()) {
            return;
        }
        if (arrayList.size() == arrayList2.size()) {
            F(arrayList, arrayList2);
            int size = arrayList.size();
            int i = 0;
            int i2 = 0;
            while (i < size) {
                if (!arrayList.get(i).p) {
                    if (i2 != i) {
                        E(arrayList, arrayList2, i2, i);
                    }
                    i2 = i + 1;
                    if (arrayList2.get(i).booleanValue()) {
                        while (i2 < size && arrayList2.get(i2).booleanValue() && !arrayList.get(i2).p) {
                            i2++;
                        }
                    }
                    E(arrayList, arrayList2, i, i2);
                    i = i2 - 1;
                }
                i++;
            }
            if (i2 != size) {
                E(arrayList, arrayList2, i2, size);
                return;
            }
            return;
        }
        throw new IllegalStateException("Internal error with the back stack records");
    }

    public void b(Fragment fragment) {
        if (N(2)) {
            Log.v("FragmentManager", "add: " + fragment);
        }
        R(fragment);
        if (fragment.mDetached) {
            return;
        }
        this.f2498c.a(fragment);
        fragment.mRemoving = false;
        if (fragment.mView == null) {
            fragment.mHiddenChanged = false;
        }
        if (O(fragment)) {
            this.s = true;
        }
    }

    public void b0(Fragment fragment) {
        if (Q()) {
            if (N(2)) {
                Log.v("FragmentManager", "Ignoring removeRetainedFragment as the state is already saved");
                return;
            }
            return;
        }
        if ((this.B.f2524d.remove(fragment.mWho) != null) && N(2)) {
            Log.v("FragmentManager", "Updating retained Fragments: Removed " + fragment);
        }
    }

    public void c(Fragment fragment) {
        boolean z;
        if (Q()) {
            if (N(2)) {
                Log.v("FragmentManager", "Ignoring addRetainedFragment as the state is already saved");
                return;
            }
            return;
        }
        u uVar = this.B;
        if (uVar.f2524d.containsKey(fragment.mWho)) {
            z = false;
        } else {
            uVar.f2524d.put(fragment.mWho, fragment);
            z = true;
        }
        if (z && N(2)) {
            Log.v("FragmentManager", "Updating retained Fragments: Added " + fragment);
        }
    }

    public void c0(Parcelable parcelable) {
        w wVar;
        if (parcelable == null) {
            return;
        }
        t tVar = (t) parcelable;
        if (tVar.f2518b == null) {
            return;
        }
        this.f2498c.f2540b.clear();
        Iterator<v> it = tVar.f2518b.iterator();
        while (it.hasNext()) {
            v next = it.next();
            if (next != null) {
                Fragment fragment = this.B.f2524d.get(next.f2530c);
                if (fragment != null) {
                    if (N(2)) {
                        Log.v("FragmentManager", "restoreSaveState: re-attaching retained " + fragment);
                    }
                    wVar = new w(this.l, fragment, next);
                } else {
                    wVar = new w(this.l, this.n.f2490c.getClassLoader(), L(), next);
                }
                Fragment fragment2 = wVar.f2537b;
                fragment2.mFragmentManager = this;
                if (N(2)) {
                    StringBuilder x = c.b.a.a.a.x("restoreSaveState: active (");
                    x.append(fragment2.mWho);
                    x.append("): ");
                    x.append(fragment2);
                    Log.v("FragmentManager", x.toString());
                }
                wVar.a(this.n.f2490c.getClassLoader());
                this.f2498c.f2540b.put(wVar.f2537b.mWho, wVar);
                wVar.f2538c = this.m;
            }
        }
        for (Fragment fragment3 : this.B.f2524d.values()) {
            if (!this.f2498c.c(fragment3.mWho)) {
                if (N(2)) {
                    Log.v("FragmentManager", "Discarding retained Fragment " + fragment3 + " that was not found in the set of active Fragments " + tVar.f2518b);
                }
                U(fragment3, 1);
                fragment3.mRemoving = true;
                U(fragment3, -1);
            }
        }
        x xVar = this.f2498c;
        ArrayList<String> arrayList = tVar.f2519c;
        xVar.f2539a.clear();
        if (arrayList != null) {
            for (String str : arrayList) {
                Fragment e2 = xVar.e(str);
                if (e2 != null) {
                    if (N(2)) {
                        Log.v("FragmentManager", "restoreSaveState: added (" + str + "): " + e2);
                    }
                    xVar.a(e2);
                } else {
                    throw new IllegalStateException(c.b.a.a.a.r("No instantiated fragment for (", str, ")"));
                }
            }
        }
        Fragment fragment4 = null;
        if (tVar.f2520d != null) {
            this.f2499d = new ArrayList<>(tVar.f2520d.length);
            int i = 0;
            while (true) {
                b.q.b.b[] bVarArr = tVar.f2520d;
                if (i >= bVarArr.length) {
                    break;
                }
                b.q.b.b bVar = bVarArr[i];
                Objects.requireNonNull(bVar);
                b.q.b.a aVar = new b.q.b.a(this);
                int i2 = 0;
                int i3 = 0;
                while (true) {
                    int[] iArr = bVar.f2398b;
                    if (i2 >= iArr.length) {
                        break;
                    }
                    y.a aVar2 = new y.a();
                    int i4 = i2 + 1;
                    aVar2.f2549a = iArr[i2];
                    if (N(2)) {
                        Log.v("FragmentManager", "Instantiate " + aVar + " op #" + i3 + " base fragment #" + bVar.f2398b[i4]);
                    }
                    String str2 = bVar.f2399c.get(i3);
                    if (str2 != null) {
                        aVar2.f2550b = this.f2498c.e(str2);
                    } else {
                        aVar2.f2550b = fragment4;
                    }
                    aVar2.f2555g = e.b.values()[bVar.f2400d[i3]];
                    aVar2.f2556h = e.b.values()[bVar.f2401e[i3]];
                    int[] iArr2 = bVar.f2398b;
                    int i5 = i4 + 1;
                    int i6 = iArr2[i4];
                    aVar2.f2551c = i6;
                    int i7 = i5 + 1;
                    int i8 = iArr2[i5];
                    aVar2.f2552d = i8;
                    int i9 = i7 + 1;
                    int i10 = iArr2[i7];
                    aVar2.f2553e = i10;
                    int i11 = iArr2[i9];
                    aVar2.f2554f = i11;
                    aVar.f2542b = i6;
                    aVar.f2543c = i8;
                    aVar.f2544d = i10;
                    aVar.f2545e = i11;
                    aVar.b(aVar2);
                    i3++;
                    fragment4 = null;
                    i2 = i9 + 1;
                }
                aVar.f2546f = bVar.f2402f;
                aVar.i = bVar.f2403g;
                aVar.s = bVar.f2404h;
                aVar.f2547g = true;
                aVar.j = bVar.i;
                aVar.k = bVar.j;
                aVar.l = bVar.k;
                aVar.m = bVar.l;
                aVar.n = bVar.m;
                aVar.o = bVar.n;
                aVar.p = bVar.o;
                aVar.e(1);
                if (N(2)) {
                    StringBuilder y = c.b.a.a.a.y("restoreAllState: back stack #", i, " (index ");
                    y.append(aVar.s);
                    y.append("): ");
                    y.append(aVar);
                    Log.v("FragmentManager", y.toString());
                    PrintWriter printWriter = new PrintWriter(new b.j.i.b("FragmentManager"));
                    aVar.i("  ", printWriter, false);
                    printWriter.close();
                }
                this.f2499d.add(aVar);
                i++;
                fragment4 = null;
            }
        } else {
            this.f2499d = null;
        }
        this.i.set(tVar.f2521e);
        String str3 = tVar.f2522f;
        if (str3 != null) {
            Fragment e3 = this.f2498c.e(str3);
            this.q = e3;
            t(e3);
        }
    }

    public void d(n<?> nVar, j jVar, Fragment fragment) {
        if (this.n == null) {
            this.n = nVar;
            this.o = jVar;
            this.p = fragment;
            if (fragment != null) {
                l0();
            }
            if (nVar instanceof b.a.c) {
                b.a.c cVar = (b.a.c) nVar;
                OnBackPressedDispatcher b2 = cVar.b();
                this.f2502g = b2;
                b.t.h hVar = cVar;
                if (fragment != null) {
                    hVar = fragment;
                }
                b2.a(hVar, this.f2503h);
            }
            if (fragment != null) {
                u uVar = fragment.mFragmentManager.B;
                u uVar2 = uVar.f2525e.get(fragment.mWho);
                if (uVar2 == null) {
                    uVar2 = new u(uVar.f2527g);
                    uVar.f2525e.put(fragment.mWho, uVar2);
                }
                this.B = uVar2;
                return;
            } else if (nVar instanceof b.t.z) {
                b.t.y viewModelStore = ((b.t.z) nVar).getViewModelStore();
                b.t.u uVar3 = u.f2523c;
                String canonicalName = u.class.getCanonicalName();
                if (canonicalName != null) {
                    String q = c.b.a.a.a.q("androidx.lifecycle.ViewModelProvider.DefaultKey:", canonicalName);
                    b.t.s sVar = viewModelStore.f2604a.get(q);
                    if (u.class.isInstance(sVar)) {
                        if (uVar3 instanceof b.t.x) {
                            ((b.t.x) uVar3).b(sVar);
                        }
                    } else {
                        sVar = uVar3 instanceof b.t.v ? ((b.t.v) uVar3).c(q, u.class) : ((u.a) uVar3).a(u.class);
                        b.t.s put = viewModelStore.f2604a.put(q, sVar);
                        if (put != null) {
                            put.a();
                        }
                    }
                    this.B = (u) sVar;
                    return;
                }
                throw new IllegalArgumentException("Local and anonymous classes can not be ViewModels");
            } else {
                this.B = new u(false);
                return;
            }
        }
        throw new IllegalStateException("Already attached");
    }

    public Parcelable d0() {
        b.q.b.b[] bVarArr;
        ArrayList<String> arrayList;
        int size;
        if (this.A != null) {
            while (!this.A.isEmpty()) {
                this.A.remove(0).a();
            }
        }
        z();
        C(true);
        this.t = true;
        x xVar = this.f2498c;
        Objects.requireNonNull(xVar);
        ArrayList<v> arrayList2 = new ArrayList<>(xVar.f2540b.size());
        Iterator<w> it = xVar.f2540b.values().iterator();
        while (true) {
            bVarArr = null;
            bVarArr = null;
            if (!it.hasNext()) {
                break;
            }
            w next = it.next();
            if (next != null) {
                Fragment fragment = next.f2537b;
                v vVar = new v(fragment);
                Fragment fragment2 = next.f2537b;
                if (fragment2.mState > -1 && vVar.n == null) {
                    Bundle bundle = new Bundle();
                    next.f2537b.performSaveInstanceState(bundle);
                    next.f2536a.j(next.f2537b, bundle, false);
                    Bundle bundle2 = bundle.isEmpty() ? null : bundle;
                    if (next.f2537b.mView != null) {
                        next.b();
                    }
                    if (next.f2537b.mSavedViewState != null) {
                        if (bundle2 == null) {
                            bundle2 = new Bundle();
                        }
                        bundle2.putSparseParcelableArray("android:view_state", next.f2537b.mSavedViewState);
                    }
                    if (!next.f2537b.mUserVisibleHint) {
                        if (bundle2 == null) {
                            bundle2 = new Bundle();
                        }
                        bundle2.putBoolean("android:user_visible_hint", next.f2537b.mUserVisibleHint);
                    }
                    vVar.n = bundle2;
                    if (next.f2537b.mTargetWho != null) {
                        if (bundle2 == null) {
                            vVar.n = new Bundle();
                        }
                        vVar.n.putString("android:target_state", next.f2537b.mTargetWho);
                        int i = next.f2537b.mTargetRequestCode;
                        if (i != 0) {
                            vVar.n.putInt("android:target_req_state", i);
                        }
                    }
                } else {
                    vVar.n = fragment2.mSavedFragmentState;
                }
                arrayList2.add(vVar);
                if (N(2)) {
                    Log.v("FragmentManager", "Saved state of " + fragment + ": " + vVar.n);
                }
            }
        }
        if (arrayList2.isEmpty()) {
            if (N(2)) {
                Log.v("FragmentManager", "saveAllState: no fragments!");
            }
            return null;
        }
        x xVar2 = this.f2498c;
        synchronized (xVar2.f2539a) {
            if (xVar2.f2539a.isEmpty()) {
                arrayList = null;
            } else {
                arrayList = new ArrayList<>(xVar2.f2539a.size());
                Iterator<Fragment> it2 = xVar2.f2539a.iterator();
                while (it2.hasNext()) {
                    Fragment next2 = it2.next();
                    arrayList.add(next2.mWho);
                    if (N(2)) {
                        Log.v("FragmentManager", "saveAllState: adding fragment (" + next2.mWho + "): " + next2);
                    }
                }
            }
        }
        ArrayList<b.q.b.a> arrayList3 = this.f2499d;
        if (arrayList3 != null && (size = arrayList3.size()) > 0) {
            bVarArr = new b.q.b.b[size];
            for (int i2 = 0; i2 < size; i2++) {
                bVarArr[i2] = new b.q.b.b(this.f2499d.get(i2));
                if (N(2)) {
                    StringBuilder y = c.b.a.a.a.y("saveAllState: adding back stack #", i2, ": ");
                    y.append(this.f2499d.get(i2));
                    Log.v("FragmentManager", y.toString());
                }
            }
        }
        t tVar = new t();
        tVar.f2518b = arrayList2;
        tVar.f2519c = arrayList;
        tVar.f2520d = bVarArr;
        tVar.f2521e = this.i.get();
        Fragment fragment3 = this.q;
        if (fragment3 != null) {
            tVar.f2522f = fragment3.mWho;
        }
        return tVar;
    }

    public void e(Fragment fragment) {
        if (N(2)) {
            Log.v("FragmentManager", "attach: " + fragment);
        }
        if (fragment.mDetached) {
            fragment.mDetached = false;
            if (fragment.mAdded) {
                return;
            }
            this.f2498c.a(fragment);
            if (N(2)) {
                Log.v("FragmentManager", "add from attach: " + fragment);
            }
            if (O(fragment)) {
                this.s = true;
            }
        }
    }

    public void e0() {
        synchronized (this.f2496a) {
            ArrayList<g> arrayList = this.A;
            boolean z = (arrayList == null || arrayList.isEmpty()) ? false : true;
            boolean z2 = this.f2496a.size() == 1;
            if (z || z2) {
                this.n.f2491d.removeCallbacks(this.C);
                this.n.f2491d.post(this.C);
                l0();
            }
        }
    }

    /* JADX DEBUG: Finally have unexpected throw blocks count: 2, expect 1 */
    public final void f(Fragment fragment) {
        HashSet<b.j.f.b> hashSet = this.j.get(fragment);
        if (hashSet != null) {
            Iterator<b.j.f.b> it = hashSet.iterator();
            while (it.hasNext()) {
                b.j.f.b next = it.next();
                synchronized (next) {
                    if (!next.f2117a) {
                        next.f2117a = true;
                        next.f2119c = true;
                        b.a aVar = next.f2118b;
                        if (aVar != null) {
                            try {
                                aVar.a();
                            } catch (Throwable th) {
                                synchronized (next) {
                                    next.f2119c = false;
                                    next.notifyAll();
                                    throw th;
                                }
                            }
                        }
                        synchronized (next) {
                            next.f2119c = false;
                            next.notifyAll();
                        }
                    }
                }
            }
            hashSet.clear();
            i(fragment);
            this.j.remove(fragment);
        }
    }

    public void f0(Fragment fragment, boolean z) {
        ViewGroup K = K(fragment);
        if (K == null || !(K instanceof k)) {
            return;
        }
        ((k) K).setDrawDisappearingViewsLast(!z);
    }

    public final void g() {
        this.f2497b = false;
        this.y.clear();
        this.x.clear();
    }

    public void g0(Fragment fragment, e.b bVar) {
        if (fragment.equals(G(fragment.mWho)) && (fragment.mHost == null || fragment.mFragmentManager == this)) {
            fragment.mMaxState = bVar;
            return;
        }
        throw new IllegalArgumentException("Fragment " + fragment + " is not an active fragment of FragmentManager " + this);
    }

    public void h(b.q.b.a aVar, boolean z, boolean z2, boolean z3) {
        if (z) {
            aVar.k(z3);
        } else {
            aVar.j();
        }
        ArrayList arrayList = new ArrayList(1);
        ArrayList arrayList2 = new ArrayList(1);
        arrayList.add(aVar);
        arrayList2.add(Boolean.valueOf(z));
        if (z2) {
            f0.p(this, arrayList, arrayList2, 0, 1, true, this.k);
        }
        if (z3) {
            T(this.m, true);
        }
        Iterator it = ((ArrayList) this.f2498c.f()).iterator();
        while (it.hasNext()) {
            Fragment fragment = (Fragment) it.next();
            if (fragment != null && fragment.mView != null && fragment.mIsNewlyAdded && aVar.l(fragment.mContainerId)) {
                float f2 = fragment.mPostponedAlpha;
                if (f2 > StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD) {
                    fragment.mView.setAlpha(f2);
                }
                if (z3) {
                    fragment.mPostponedAlpha = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
                } else {
                    fragment.mPostponedAlpha = -1.0f;
                    fragment.mIsNewlyAdded = false;
                }
            }
        }
    }

    public void h0(Fragment fragment) {
        if (fragment != null && (!fragment.equals(G(fragment.mWho)) || (fragment.mHost != null && fragment.mFragmentManager != this))) {
            throw new IllegalArgumentException("Fragment " + fragment + " is not an active fragment of FragmentManager " + this);
        }
        Fragment fragment2 = this.q;
        this.q = fragment;
        t(fragment2);
        t(this.q);
    }

    public final void i(Fragment fragment) {
        fragment.performDestroyView();
        this.l.n(fragment, false);
        fragment.mContainer = null;
        fragment.mView = null;
        fragment.mViewLifecycleOwner = null;
        fragment.mViewLifecycleOwnerLiveData.h(null);
        fragment.mInLayout = false;
    }

    public final void i0(Fragment fragment) {
        ViewGroup K = K(fragment);
        if (K != null) {
            if (K.getTag(R.id.visible_removing_fragment_view_tag) == null) {
                K.setTag(R.id.visible_removing_fragment_view_tag, fragment);
            }
            ((Fragment) K.getTag(R.id.visible_removing_fragment_view_tag)).setNextAnim(fragment.getNextAnim());
        }
    }

    public void j(Fragment fragment) {
        if (N(2)) {
            Log.v("FragmentManager", "detach: " + fragment);
        }
        if (fragment.mDetached) {
            return;
        }
        fragment.mDetached = true;
        if (fragment.mAdded) {
            if (N(2)) {
                Log.v("FragmentManager", "remove from detach: " + fragment);
            }
            this.f2498c.h(fragment);
            if (O(fragment)) {
                this.s = true;
            }
            i0(fragment);
        }
    }

    public void j0(Fragment fragment) {
        if (N(2)) {
            Log.v("FragmentManager", "show: " + fragment);
        }
        if (fragment.mHidden) {
            fragment.mHidden = false;
            fragment.mHiddenChanged = !fragment.mHiddenChanged;
        }
    }

    public void k(Configuration configuration) {
        for (Fragment fragment : this.f2498c.g()) {
            if (fragment != null) {
                fragment.performConfigurationChanged(configuration);
            }
        }
    }

    public final void k0() {
        Iterator it = ((ArrayList) this.f2498c.f()).iterator();
        while (it.hasNext()) {
            Fragment fragment = (Fragment) it.next();
            if (fragment != null) {
                W(fragment);
            }
        }
    }

    public boolean l(MenuItem menuItem) {
        if (this.m < 1) {
            return false;
        }
        for (Fragment fragment : this.f2498c.g()) {
            if (fragment != null && fragment.performContextItemSelected(menuItem)) {
                return true;
            }
        }
        return false;
    }

    public final void l0() {
        synchronized (this.f2496a) {
            boolean z = true;
            if (!this.f2496a.isEmpty()) {
                this.f2503h.f530a = true;
                return;
            }
            b.a.b bVar = this.f2503h;
            ArrayList<b.q.b.a> arrayList = this.f2499d;
            if ((arrayList != null ? arrayList.size() : 0) <= 0 || !P(this.p)) {
                z = false;
            }
            bVar.f530a = z;
        }
    }

    public void m() {
        this.t = false;
        this.u = false;
        w(1);
    }

    public boolean n(Menu menu, MenuInflater menuInflater) {
        if (this.m < 1) {
            return false;
        }
        ArrayList<Fragment> arrayList = null;
        boolean z = false;
        for (Fragment fragment : this.f2498c.g()) {
            if (fragment != null && fragment.performCreateOptionsMenu(menu, menuInflater)) {
                if (arrayList == null) {
                    arrayList = new ArrayList<>();
                }
                arrayList.add(fragment);
                z = true;
            }
        }
        if (this.f2500e != null) {
            for (int i = 0; i < this.f2500e.size(); i++) {
                Fragment fragment2 = this.f2500e.get(i);
                if (arrayList == null || !arrayList.contains(fragment2)) {
                    fragment2.onDestroyOptionsMenu();
                }
            }
        }
        this.f2500e = arrayList;
        return z;
    }

    public void o() {
        this.v = true;
        C(true);
        z();
        w(-1);
        this.n = null;
        this.o = null;
        this.p = null;
        if (this.f2502g != null) {
            this.f2503h.b();
            this.f2502g = null;
        }
    }

    public void p() {
        for (Fragment fragment : this.f2498c.g()) {
            if (fragment != null) {
                fragment.performLowMemory();
            }
        }
    }

    public void q(boolean z) {
        for (Fragment fragment : this.f2498c.g()) {
            if (fragment != null) {
                fragment.performMultiWindowModeChanged(z);
            }
        }
    }

    public boolean r(MenuItem menuItem) {
        if (this.m < 1) {
            return false;
        }
        for (Fragment fragment : this.f2498c.g()) {
            if (fragment != null && fragment.performOptionsItemSelected(menuItem)) {
                return true;
            }
        }
        return false;
    }

    public void s(Menu menu) {
        if (this.m < 1) {
            return;
        }
        for (Fragment fragment : this.f2498c.g()) {
            if (fragment != null) {
                fragment.performOptionsMenuClosed(menu);
            }
        }
    }

    public final void t(Fragment fragment) {
        if (fragment == null || !fragment.equals(G(fragment.mWho))) {
            return;
        }
        fragment.performPrimaryNavigationFragmentChanged();
    }

    public String toString() {
        StringBuilder sb = new StringBuilder(128);
        sb.append("FragmentManager{");
        sb.append(Integer.toHexString(System.identityHashCode(this)));
        sb.append(" in ");
        Fragment fragment = this.p;
        if (fragment != null) {
            sb.append(fragment.getClass().getSimpleName());
            sb.append("{");
            sb.append(Integer.toHexString(System.identityHashCode(this.p)));
            sb.append("}");
        } else {
            n<?> nVar = this.n;
            if (nVar != null) {
                sb.append(nVar.getClass().getSimpleName());
                sb.append("{");
                sb.append(Integer.toHexString(System.identityHashCode(this.n)));
                sb.append("}");
            } else {
                sb.append("null");
            }
        }
        sb.append("}}");
        return sb.toString();
    }

    public void u(boolean z) {
        for (Fragment fragment : this.f2498c.g()) {
            if (fragment != null) {
                fragment.performPictureInPictureModeChanged(z);
            }
        }
    }

    public boolean v(Menu menu) {
        boolean z = false;
        if (this.m < 1) {
            return false;
        }
        for (Fragment fragment : this.f2498c.g()) {
            if (fragment != null && fragment.performPrepareOptionsMenu(menu)) {
                z = true;
            }
        }
        return z;
    }

    public final void w(int i) {
        try {
            this.f2497b = true;
            this.f2498c.d(i);
            T(i, false);
            this.f2497b = false;
            C(true);
        } catch (Throwable th) {
            this.f2497b = false;
            throw th;
        }
    }

    public final void x() {
        if (this.w) {
            this.w = false;
            k0();
        }
    }

    public void y(String str, FileDescriptor fileDescriptor, PrintWriter printWriter, String[] strArr) {
        int size;
        int size2;
        String q = c.b.a.a.a.q(str, "    ");
        x xVar = this.f2498c;
        Objects.requireNonNull(xVar);
        String str2 = str + "    ";
        if (!xVar.f2540b.isEmpty()) {
            printWriter.print(str);
            printWriter.print("Active Fragments:");
            for (w wVar : xVar.f2540b.values()) {
                printWriter.print(str);
                if (wVar != null) {
                    Fragment fragment = wVar.f2537b;
                    printWriter.println(fragment);
                    fragment.dump(str2, fileDescriptor, printWriter, strArr);
                } else {
                    printWriter.println("null");
                }
            }
        }
        int size3 = xVar.f2539a.size();
        if (size3 > 0) {
            printWriter.print(str);
            printWriter.println("Added Fragments:");
            for (int i = 0; i < size3; i++) {
                printWriter.print(str);
                printWriter.print("  #");
                printWriter.print(i);
                printWriter.print(": ");
                printWriter.println(xVar.f2539a.get(i).toString());
            }
        }
        ArrayList<Fragment> arrayList = this.f2500e;
        if (arrayList != null && (size2 = arrayList.size()) > 0) {
            printWriter.print(str);
            printWriter.println("Fragments Created Menus:");
            for (int i2 = 0; i2 < size2; i2++) {
                printWriter.print(str);
                printWriter.print("  #");
                printWriter.print(i2);
                printWriter.print(": ");
                printWriter.println(this.f2500e.get(i2).toString());
            }
        }
        ArrayList<b.q.b.a> arrayList2 = this.f2499d;
        if (arrayList2 != null && (size = arrayList2.size()) > 0) {
            printWriter.print(str);
            printWriter.println("Back Stack:");
            for (int i3 = 0; i3 < size; i3++) {
                b.q.b.a aVar = this.f2499d.get(i3);
                printWriter.print(str);
                printWriter.print("  #");
                printWriter.print(i3);
                printWriter.print(": ");
                printWriter.println(aVar.toString());
                aVar.i(q, printWriter, true);
            }
        }
        printWriter.print(str);
        printWriter.println("Back Stack Index: " + this.i.get());
        synchronized (this.f2496a) {
            int size4 = this.f2496a.size();
            if (size4 > 0) {
                printWriter.print(str);
                printWriter.println("Pending Actions:");
                for (int i4 = 0; i4 < size4; i4++) {
                    printWriter.print(str);
                    printWriter.print("  #");
                    printWriter.print(i4);
                    printWriter.print(": ");
                    printWriter.println((e) this.f2496a.get(i4));
                }
            }
        }
        printWriter.print(str);
        printWriter.println("FragmentManager misc state:");
        printWriter.print(str);
        printWriter.print("  mHost=");
        printWriter.println(this.n);
        printWriter.print(str);
        printWriter.print("  mContainer=");
        printWriter.println(this.o);
        if (this.p != null) {
            printWriter.print(str);
            printWriter.print("  mParent=");
            printWriter.println(this.p);
        }
        printWriter.print(str);
        printWriter.print("  mCurState=");
        printWriter.print(this.m);
        printWriter.print(" mStateSaved=");
        printWriter.print(this.t);
        printWriter.print(" mStopped=");
        printWriter.print(this.u);
        printWriter.print(" mDestroyed=");
        printWriter.println(this.v);
        if (this.s) {
            printWriter.print(str);
            printWriter.print("  mNeedMenuInvalidate=");
            printWriter.println(this.s);
        }
    }

    public final void z() {
        if (this.j.isEmpty()) {
            return;
        }
        for (Fragment fragment : this.j.keySet()) {
            f(fragment);
            U(fragment, fragment.getStateAfterAnimating());
        }
    }
}