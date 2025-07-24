package b.v.u;

import android.content.Context;
import android.content.res.TypedArray;
import android.os.Bundle;
import android.util.AttributeSet;
import android.util.Log;
import android.view.View;
import androidx.fragment.app.Fragment;
import b.q.b.f0;
import b.q.b.q;
import b.v.j;
import b.v.o;
import b.v.q;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicInteger;

/* compiled from: FragmentNavigator.java */
@q.b("fragment")
/* loaded from: classes.dex */
public class a extends q<C0052a> {

    /* renamed from: a  reason: collision with root package name */
    public final Context f2687a;

    /* renamed from: b  reason: collision with root package name */
    public final b.q.b.q f2688b;

    /* renamed from: c  reason: collision with root package name */
    public final int f2689c;

    /* renamed from: d  reason: collision with root package name */
    public ArrayDeque<Integer> f2690d = new ArrayDeque<>();

    /* compiled from: FragmentNavigator.java */
    /* renamed from: b.v.u.a$a  reason: collision with other inner class name */
    /* loaded from: classes.dex */
    public static class C0052a extends j {
        public String j;

        public C0052a(q<? extends C0052a> qVar) {
            super(qVar);
        }

        @Override // b.v.j
        public void d(Context context, AttributeSet attributeSet) {
            super.d(context, attributeSet);
            TypedArray obtainAttributes = context.getResources().obtainAttributes(attributeSet, d.f2699b);
            String string = obtainAttributes.getString(0);
            if (string != null) {
                this.j = string;
            }
            obtainAttributes.recycle();
        }

        @Override // b.v.j
        public String toString() {
            StringBuilder sb = new StringBuilder();
            sb.append(super.toString());
            sb.append(" class=");
            String str = this.j;
            if (str == null) {
                sb.append("null");
            } else {
                sb.append(str);
            }
            return sb.toString();
        }
    }

    /* compiled from: FragmentNavigator.java */
    /* loaded from: classes.dex */
    public static final class b implements q.a {
    }

    public a(Context context, b.q.b.q qVar, int i) {
        this.f2687a = context;
        this.f2688b = qVar;
        this.f2689c = i;
    }

    /* JADX DEBUG: Return type fixed from 'b.v.j' to match base method */
    @Override // b.v.q
    public C0052a a() {
        return new C0052a(this);
    }

    /* JADX DEBUG: Method arguments types fixed to match base method, original types: [b.v.j, android.os.Bundle, b.v.o, b.v.q$a] */
    /* JADX WARN: Removed duplicated region for block: B:68:0x0116  */
    /* JADX WARN: Removed duplicated region for block: B:89:0x01a0  */
    @Override // b.v.q
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public j b(C0052a c0052a, Bundle bundle, o oVar, q.a aVar) {
        C0052a c0052a2 = c0052a;
        if (this.f2688b.Q()) {
            Log.i("FragmentNavigator", "Ignoring navigate() call: FragmentManager has already saved its state");
        } else {
            String str = c0052a2.j;
            if (str != null) {
                boolean z = false;
                if (str.charAt(0) == '.') {
                    str = this.f2687a.getPackageName() + str;
                }
                Fragment a2 = this.f2688b.L().a(this.f2687a.getClassLoader(), str);
                a2.setArguments(bundle);
                b.q.b.a aVar2 = new b.q.b.a(this.f2688b);
                int i = oVar != null ? oVar.f2665d : -1;
                int i2 = oVar != null ? oVar.f2666e : -1;
                int i3 = oVar != null ? oVar.f2667f : -1;
                int i4 = oVar != null ? oVar.f2668g : -1;
                if (i != -1 || i2 != -1 || i3 != -1 || i4 != -1) {
                    if (i == -1) {
                        i = 0;
                    }
                    if (i2 == -1) {
                        i2 = 0;
                    }
                    if (i3 == -1) {
                        i3 = 0;
                    }
                    if (i4 == -1) {
                        i4 = 0;
                    }
                    aVar2.f2542b = i;
                    aVar2.f2543c = i2;
                    aVar2.f2544d = i3;
                    aVar2.f2545e = i4;
                }
                int i5 = this.f2689c;
                if (i5 != 0) {
                    aVar2.d(i5, a2, null, 2);
                    aVar2.o(a2);
                    int i6 = c0052a2.f2645d;
                    boolean isEmpty = this.f2690d.isEmpty();
                    boolean z2 = oVar != null && !isEmpty && oVar.f2662a && this.f2690d.peekLast().intValue() == i6;
                    if (!isEmpty) {
                        if (z2) {
                            if (this.f2690d.size() > 1) {
                                b.q.b.q qVar = this.f2688b;
                                qVar.A(new q.f(f(this.f2690d.size(), this.f2690d.peekLast().intValue()), -1, 1), false);
                                String f2 = f(this.f2690d.size(), i6);
                                if (aVar2.f2548h) {
                                    aVar2.f2547g = true;
                                    aVar2.i = f2;
                                } else {
                                    throw new IllegalStateException("This FragmentTransaction is not allowed to be added to the back stack.");
                                }
                            }
                            if (aVar instanceof b) {
                                Objects.requireNonNull((b) aVar);
                                for (Map.Entry entry : Collections.unmodifiableMap(null).entrySet()) {
                                    String str2 = (String) entry.getValue();
                                    int[] iArr = f0.f2441a;
                                    AtomicInteger atomicInteger = b.j.j.q.f2214a;
                                    String transitionName = ((View) entry.getKey()).getTransitionName();
                                    if (transitionName != null) {
                                        if (aVar2.n == null) {
                                            aVar2.n = new ArrayList<>();
                                            aVar2.o = new ArrayList<>();
                                        } else if (!aVar2.o.contains(str2)) {
                                            if (aVar2.n.contains(transitionName)) {
                                                throw new IllegalArgumentException(c.b.a.a.a.r("A shared element with the source name '", transitionName, "' has already been added to the transaction."));
                                            }
                                        } else {
                                            throw new IllegalArgumentException(c.b.a.a.a.r("A shared element with the target name '", str2, "' has already been added to the transaction."));
                                        }
                                        aVar2.n.add(transitionName);
                                        aVar2.o.add(str2);
                                    } else {
                                        throw new IllegalArgumentException("Unique transitionNames are required for all sharedElements");
                                    }
                                }
                            }
                            aVar2.p = true;
                            aVar2.c();
                            if (z) {
                                this.f2690d.add(Integer.valueOf(i6));
                                return c0052a2;
                            }
                        } else {
                            String f3 = f(this.f2690d.size() + 1, i6);
                            if (aVar2.f2548h) {
                                aVar2.f2547g = true;
                                aVar2.i = f3;
                            } else {
                                throw new IllegalStateException("This FragmentTransaction is not allowed to be added to the back stack.");
                            }
                        }
                    }
                    z = true;
                    if (aVar instanceof b) {
                    }
                    aVar2.p = true;
                    aVar2.c();
                    if (z) {
                    }
                } else {
                    throw new IllegalArgumentException("Must use non-zero containerViewId");
                }
            } else {
                throw new IllegalStateException("Fragment class was not set");
            }
        }
        return null;
    }

    @Override // b.v.q
    public void c(Bundle bundle) {
        int[] intArray = bundle.getIntArray("androidx-nav-fragment:navigator:backStackIds");
        if (intArray != null) {
            this.f2690d.clear();
            for (int i : intArray) {
                this.f2690d.add(Integer.valueOf(i));
            }
        }
    }

    @Override // b.v.q
    public Bundle d() {
        Bundle bundle = new Bundle();
        int[] iArr = new int[this.f2690d.size()];
        Iterator<Integer> it = this.f2690d.iterator();
        int i = 0;
        while (it.hasNext()) {
            iArr[i] = it.next().intValue();
            i++;
        }
        bundle.putIntArray("androidx-nav-fragment:navigator:backStackIds", iArr);
        return bundle;
    }

    @Override // b.v.q
    public boolean e() {
        if (this.f2690d.isEmpty()) {
            return false;
        }
        if (this.f2688b.Q()) {
            Log.i("FragmentNavigator", "Ignoring popBackStack() call: FragmentManager has already saved its state");
            return false;
        }
        b.q.b.q qVar = this.f2688b;
        qVar.A(new q.f(f(this.f2690d.size(), this.f2690d.peekLast().intValue()), -1, 1), false);
        this.f2690d.removeLast();
        return true;
    }

    public final String f(int i, int i2) {
        return i + "-" + i2;
    }
}