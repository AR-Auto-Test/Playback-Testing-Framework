package androidx.navigation;

import android.app.Activity;
import android.content.ComponentName;
import android.content.Context;
import android.content.ContextWrapper;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.os.Parcelable;
import android.util.Log;
import b.j.b.d;
import b.t.e;
import b.t.f;
import b.t.h;
import b.t.y;
import b.v.e;
import b.v.g;
import b.v.i;
import b.v.j;
import b.v.k;
import b.v.l;
import b.v.n;
import b.v.o;
import b.v.q;
import b.v.r;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Objects;
import java.util.concurrent.CopyOnWriteArrayList;
import org.opencv.calib3d.Calib3d;

/* loaded from: classes.dex */
public class NavController {

    /* renamed from: a  reason: collision with root package name */
    public final Context f342a;

    /* renamed from: b  reason: collision with root package name */
    public Activity f343b;

    /* renamed from: c  reason: collision with root package name */
    public n f344c;

    /* renamed from: d  reason: collision with root package name */
    public k f345d;

    /* renamed from: e  reason: collision with root package name */
    public Bundle f346e;

    /* renamed from: f  reason: collision with root package name */
    public Parcelable[] f347f;

    /* renamed from: g  reason: collision with root package name */
    public boolean f348g;
    public h i;
    public g j;

    /* renamed from: h  reason: collision with root package name */
    public final Deque<e> f349h = new ArrayDeque();
    public r k = new r();
    public final CopyOnWriteArrayList<b> l = new CopyOnWriteArrayList<>();
    public final b.t.g m = new f() { // from class: androidx.navigation.NavController.1
        @Override // b.t.f
        public void e(h hVar, e.a aVar) {
            e.b bVar;
            NavController navController = NavController.this;
            if (navController.f345d != null) {
                for (b.v.e eVar : navController.f349h) {
                    Objects.requireNonNull(eVar);
                    int ordinal = aVar.ordinal();
                    if (ordinal != 0) {
                        if (ordinal != 1) {
                            if (ordinal == 2) {
                                bVar = e.b.RESUMED;
                            } else if (ordinal != 3) {
                                if (ordinal != 4) {
                                    if (ordinal == 5) {
                                        bVar = e.b.DESTROYED;
                                    } else {
                                        throw new IllegalArgumentException("Unexpected event value " + aVar);
                                    }
                                }
                            }
                            eVar.f2620g = bVar;
                            eVar.a();
                        }
                        bVar = e.b.STARTED;
                        eVar.f2620g = bVar;
                        eVar.a();
                    }
                    bVar = e.b.CREATED;
                    eVar.f2620g = bVar;
                    eVar.a();
                }
            }
        }
    };
    public final b.a.b n = new a(false);
    public boolean o = true;

    /* loaded from: classes.dex */
    public class a extends b.a.b {
        public a(boolean z) {
            super(z);
        }

        @Override // b.a.b
        public void a() {
            NavController.this.e();
        }
    }

    /* loaded from: classes.dex */
    public interface b {
        void a(NavController navController, j jVar, Bundle bundle);
    }

    public NavController(Context context) {
        this.f342a = context;
        while (true) {
            if (!(context instanceof ContextWrapper)) {
                break;
            } else if (context instanceof Activity) {
                this.f343b = (Activity) context;
                break;
            } else {
                context = ((ContextWrapper) context).getBaseContext();
            }
        }
        r rVar = this.k;
        rVar.a(new l(rVar));
        this.k.a(new b.v.a(this.f342a));
    }

    public final boolean a() {
        e.b bVar = e.b.STARTED;
        e.b bVar2 = e.b.RESUMED;
        while (!this.f349h.isEmpty() && (this.f349h.peekLast().f2615b instanceof k) && f(this.f349h.peekLast().f2615b.f2645d, true)) {
        }
        if (this.f349h.isEmpty()) {
            return false;
        }
        j jVar = this.f349h.peekLast().f2615b;
        j jVar2 = null;
        if (jVar instanceof b.v.b) {
            Iterator<b.v.e> descendingIterator = this.f349h.descendingIterator();
            while (true) {
                if (!descendingIterator.hasNext()) {
                    break;
                }
                j jVar3 = descendingIterator.next().f2615b;
                if (!(jVar3 instanceof k) && !(jVar3 instanceof b.v.b)) {
                    jVar2 = jVar3;
                    break;
                }
            }
        }
        HashMap hashMap = new HashMap();
        Iterator<b.v.e> descendingIterator2 = this.f349h.descendingIterator();
        while (descendingIterator2.hasNext()) {
            b.v.e next = descendingIterator2.next();
            e.b bVar3 = next.f2621h;
            j jVar4 = next.f2615b;
            if (jVar != null && jVar4.f2645d == jVar.f2645d) {
                if (bVar3 != bVar2) {
                    hashMap.put(next, bVar2);
                }
                jVar = jVar.f2644c;
            } else if (jVar2 != null && jVar4.f2645d == jVar2.f2645d) {
                if (bVar3 == bVar2) {
                    next.f2621h = bVar;
                    next.a();
                } else if (bVar3 != bVar) {
                    hashMap.put(next, bVar);
                }
                jVar2 = jVar2.f2644c;
            } else {
                next.f2621h = e.b.CREATED;
                next.a();
            }
        }
        for (b.v.e eVar : this.f349h) {
            e.b bVar4 = (e.b) hashMap.get(eVar);
            if (bVar4 != null) {
                eVar.f2621h = bVar4;
                eVar.a();
            } else {
                eVar.a();
            }
        }
        b.v.e peekLast = this.f349h.peekLast();
        Iterator<b> it = this.l.iterator();
        while (it.hasNext()) {
            it.next().a(this, peekLast.f2615b, peekLast.f2616c);
        }
        return true;
    }

    public j b(int i) {
        j jVar;
        k kVar;
        k kVar2 = this.f345d;
        if (kVar2 == null) {
            return null;
        }
        if (kVar2.f2645d == i) {
            return kVar2;
        }
        if (this.f349h.isEmpty()) {
            jVar = this.f345d;
        } else {
            jVar = this.f349h.getLast().f2615b;
        }
        if (jVar instanceof k) {
            kVar = (k) jVar;
        } else {
            kVar = jVar.f2644c;
        }
        return kVar.g(i, true);
    }

    public j c() {
        b.v.e last = this.f349h.isEmpty() ? null : this.f349h.getLast();
        if (last != null) {
            return last.f2615b;
        }
        return null;
    }

    public final void d(j jVar, Bundle bundle, o oVar, q.a aVar) {
        int i;
        boolean z = false;
        boolean f2 = (oVar == null || (i = oVar.f2663b) == -1) ? false : f(i, oVar.f2664c);
        q c2 = this.k.c(jVar.f2643b);
        Bundle a2 = jVar.a(bundle);
        j b2 = c2.b(jVar, a2, oVar, null);
        if (b2 != null) {
            if (!(b2 instanceof b.v.b)) {
                while (!this.f349h.isEmpty() && (this.f349h.peekLast().f2615b instanceof b.v.b) && f(this.f349h.peekLast().f2615b.f2645d, true)) {
                }
            }
            ArrayDeque arrayDeque = new ArrayDeque();
            if (jVar instanceof k) {
                k kVar = b2;
                while (true) {
                    k kVar2 = kVar.f2644c;
                    if (kVar2 != null) {
                        arrayDeque.addFirst(new b.v.e(this.f342a, kVar2, a2, this.i, this.j));
                        if (!this.f349h.isEmpty() && this.f349h.getLast().f2615b == kVar2) {
                            f(kVar2.f2645d, true);
                        }
                    }
                    if (kVar2 == null || kVar2 == jVar) {
                        break;
                    }
                    kVar = kVar2;
                }
            }
            j jVar2 = arrayDeque.isEmpty() ? b2 : ((b.v.e) arrayDeque.getFirst()).f2615b;
            while (jVar2 != null && b(jVar2.f2645d) == null) {
                jVar2 = jVar2.f2644c;
                if (jVar2 != null) {
                    arrayDeque.addFirst(new b.v.e(this.f342a, jVar2, a2, this.i, this.j));
                }
            }
            j jVar3 = arrayDeque.isEmpty() ? b2 : ((b.v.e) arrayDeque.getLast()).f2615b;
            while (!this.f349h.isEmpty() && (this.f349h.getLast().f2615b instanceof k) && ((k) this.f349h.getLast().f2615b).g(jVar3.f2645d, false) == null && f(this.f349h.getLast().f2615b.f2645d, true)) {
            }
            this.f349h.addAll(arrayDeque);
            if (this.f349h.isEmpty() || this.f349h.getFirst().f2615b != this.f345d) {
                this.f349h.addFirst(new b.v.e(this.f342a, this.f345d, a2, this.i, this.j));
            }
            this.f349h.add(new b.v.e(this.f342a, b2, b2.a(a2), this.i, this.j));
        } else if (oVar != null && oVar.f2662a) {
            b.v.e peekLast = this.f349h.peekLast();
            if (peekLast != null) {
                peekLast.f2616c = a2;
            }
            z = true;
        }
        h();
        if (f2 || b2 != null || z) {
            a();
        }
    }

    public boolean e() {
        return !this.f349h.isEmpty() && f(c().f2645d, true) && a();
    }

    public boolean f(int i, boolean z) {
        boolean z2;
        if (this.f349h.isEmpty()) {
            return false;
        }
        ArrayList arrayList = new ArrayList();
        Iterator<b.v.e> descendingIterator = this.f349h.descendingIterator();
        while (true) {
            if (!descendingIterator.hasNext()) {
                z2 = false;
                break;
            }
            j jVar = descendingIterator.next().f2615b;
            q c2 = this.k.c(jVar.f2643b);
            if (z || jVar.f2645d != i) {
                arrayList.add(c2);
            }
            if (jVar.f2645d == i) {
                z2 = true;
                break;
            }
        }
        if (!z2) {
            Log.i("NavController", "Ignoring popBackStack to destination " + j.b(this.f342a, i) + " as it was not found on the current back stack");
            return false;
        }
        Iterator it = arrayList.iterator();
        boolean z3 = false;
        while (it.hasNext() && ((q) it.next()).e()) {
            b.v.e removeLast = this.f349h.removeLast();
            if (removeLast.f2617d.f2579b.compareTo(e.b.CREATED) >= 0) {
                removeLast.f2621h = e.b.DESTROYED;
                removeLast.a();
            }
            g gVar = this.j;
            if (gVar != null) {
                y remove = gVar.f2627d.remove(removeLast.f2619f);
                if (remove != null) {
                    remove.a();
                }
            }
            z3 = true;
        }
        h();
        return z3;
    }

    /* JADX WARN: Code restructure failed: missing block: B:157:0x0315, code lost:
        if (r0 == false) goto L182;
     */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public void g(int i, Bundle bundle) {
        Activity activity;
        boolean z;
        j.a c2;
        String str;
        k kVar;
        j f2;
        k kVar2;
        ArrayList<String> stringArrayList;
        if (this.f344c == null) {
            this.f344c = new n(this.f342a, this.k);
        }
        k c3 = this.f344c.c(i);
        k kVar3 = this.f345d;
        boolean z2 = true;
        if (kVar3 != null) {
            f(kVar3.f2645d, true);
        }
        this.f345d = c3;
        Bundle bundle2 = this.f346e;
        if (bundle2 != null && (stringArrayList = bundle2.getStringArrayList("android-support-nav:controller:navigatorState:names")) != null) {
            Iterator<String> it = stringArrayList.iterator();
            while (it.hasNext()) {
                String next = it.next();
                q c4 = this.k.c(next);
                Bundle bundle3 = this.f346e.getBundle(next);
                if (bundle3 != null) {
                    c4.c(bundle3);
                }
            }
        }
        Parcelable[] parcelableArr = this.f347f;
        if (parcelableArr != null) {
            for (Parcelable parcelable : parcelableArr) {
                b.v.f fVar = (b.v.f) parcelable;
                j b2 = b(fVar.f2623c);
                if (b2 != null) {
                    Bundle bundle4 = fVar.f2624d;
                    if (bundle4 != null) {
                        bundle4.setClassLoader(this.f342a.getClassLoader());
                    }
                    this.f349h.add(new b.v.e(this.f342a, b2, bundle4, this.i, this.j, fVar.f2622b, fVar.f2625e));
                } else {
                    StringBuilder B = c.b.a.a.a.B("Restoring the Navigation back stack failed: destination ", j.b(this.f342a, fVar.f2623c), " cannot be found from the current destination ");
                    B.append(c());
                    throw new IllegalStateException(B.toString());
                }
            }
            h();
            this.f347f = null;
        }
        if (this.f345d != null && this.f349h.isEmpty()) {
            if (!this.f348g && (activity = this.f343b) != null) {
                Intent intent = activity.getIntent();
                if (intent != null) {
                    Bundle extras = intent.getExtras();
                    int[] intArray = extras != null ? extras.getIntArray("android-support-nav:controller:deepLinkIds") : null;
                    Bundle bundle5 = new Bundle();
                    Bundle bundle6 = extras != null ? extras.getBundle("android-support-nav:controller:deepLinkExtras") : null;
                    if (bundle6 != null) {
                        bundle5.putAll(bundle6);
                    }
                    if ((intArray == null || intArray.length == 0) && intent.getData() != null && (c2 = this.f345d.c(new i(intent))) != null) {
                        j jVar = c2.f2650b;
                        Objects.requireNonNull(jVar);
                        ArrayDeque arrayDeque = new ArrayDeque();
                        k kVar4 = jVar;
                        while (true) {
                            k kVar5 = kVar4.f2644c;
                            if (kVar5 == null || kVar5.k != kVar4.f2645d) {
                                arrayDeque.addFirst(kVar4);
                            }
                            if (kVar5 == null) {
                                break;
                            }
                            kVar4 = kVar5;
                        }
                        intArray = new int[arrayDeque.size()];
                        Iterator it2 = arrayDeque.iterator();
                        int i2 = 0;
                        while (it2.hasNext()) {
                            intArray[i2] = ((j) it2.next()).f2645d;
                            i2++;
                        }
                        bundle5.putAll(jVar.a(c2.f2651c));
                    }
                    if (intArray != null && intArray.length != 0) {
                        k kVar6 = this.f345d;
                        int i3 = 0;
                        while (true) {
                            if (i3 >= intArray.length) {
                                str = null;
                                break;
                            }
                            int i4 = intArray[i3];
                            if (i3 == 0) {
                                f2 = this.f345d;
                                if (f2.f2645d != i4) {
                                    f2 = null;
                                }
                            } else {
                                f2 = kVar6.f(i4);
                            }
                            if (f2 == null) {
                                str = j.b(this.f342a, i4);
                                break;
                            }
                            if (i3 != intArray.length - 1) {
                                while (true) {
                                    kVar2 = (k) f2;
                                    if (!(kVar2.f(kVar2.k) instanceof k)) {
                                        break;
                                    }
                                    f2 = kVar2.f(kVar2.k);
                                }
                                kVar6 = kVar2;
                            }
                            i3++;
                        }
                        if (str != null) {
                            Log.i("NavController", "Could not find destination " + str + " in the navigation graph, ignoring the deep link from " + intent);
                        } else {
                            bundle5.putParcelable("android-support-nav:controller:deepLinkIntent", intent);
                            int flags = intent.getFlags();
                            int i5 = 268435456 & flags;
                            if (i5 != 0 && (flags & Calib3d.CALIB_THIN_PRISM_MODEL) == 0) {
                                intent.addFlags(Calib3d.CALIB_THIN_PRISM_MODEL);
                                Context context = this.f342a;
                                ArrayList arrayList = new ArrayList();
                                ComponentName component = intent.getComponent();
                                if (component == null) {
                                    component = intent.resolveActivity(context.getPackageManager());
                                }
                                if (component != null) {
                                    int size = arrayList.size();
                                    try {
                                        for (Intent v = d.v(context, component); v != null; v = d.v(context, v.getComponent())) {
                                            arrayList.add(size, v);
                                        }
                                    } catch (PackageManager.NameNotFoundException e2) {
                                        Log.e("TaskStackBuilder", "Bad ComponentName while traversing activity parent metadata");
                                        throw new IllegalArgumentException(e2);
                                    }
                                }
                                arrayList.add(intent);
                                if (!arrayList.isEmpty()) {
                                    Intent[] intentArr = (Intent[]) arrayList.toArray(new Intent[arrayList.size()]);
                                    intentArr[0] = new Intent(intentArr[0]).addFlags(268484608);
                                    Object obj = b.j.c.a.f2074a;
                                    context.startActivities(intentArr, null);
                                    Activity activity2 = this.f343b;
                                    if (activity2 != null) {
                                        activity2.finish();
                                        this.f343b.overridePendingTransition(0, 0);
                                    }
                                } else {
                                    throw new IllegalStateException("No intents added to TaskStackBuilder; cannot startActivities");
                                }
                            } else if (i5 != 0) {
                                if (!this.f349h.isEmpty()) {
                                    f(this.f345d.f2645d, true);
                                }
                                int i6 = 0;
                                while (i6 < intArray.length) {
                                    int i7 = i6 + 1;
                                    int i8 = intArray[i6];
                                    j b3 = b(i8);
                                    if (b3 != null) {
                                        d(b3, bundle5, new o(false, -1, false, 0, 0, -1, -1), null);
                                        i6 = i7;
                                    } else {
                                        StringBuilder B2 = c.b.a.a.a.B("Deep Linking failed: destination ", j.b(this.f342a, i8), " cannot be found from the current destination ");
                                        B2.append(c());
                                        throw new IllegalStateException(B2.toString());
                                    }
                                }
                            } else {
                                k kVar7 = this.f345d;
                                int i9 = 0;
                                while (i9 < intArray.length) {
                                    int i10 = intArray[i9];
                                    j f3 = i9 == 0 ? this.f345d : kVar7.f(i10);
                                    if (f3 != null) {
                                        if (i9 != intArray.length - 1) {
                                            while (true) {
                                                kVar = (k) f3;
                                                if (!(kVar.f(kVar.k) instanceof k)) {
                                                    break;
                                                }
                                                f3 = kVar.f(kVar.k);
                                            }
                                            kVar7 = kVar;
                                        } else {
                                            d(f3, f3.a(bundle5), new o(false, this.f345d.f2645d, true, 0, 0, -1, -1), null);
                                        }
                                        i9++;
                                    } else {
                                        throw new IllegalStateException("Deep Linking failed: destination " + j.b(this.f342a, i10) + " cannot be found in graph " + kVar7);
                                    }
                                }
                                this.f348g = true;
                            }
                            z = true;
                        }
                    }
                }
                z = false;
            }
            z2 = false;
            if (z2) {
                return;
            }
            d(this.f345d, bundle, null, null);
            return;
        }
        a();
    }

    public final void h() {
        b.a.b bVar = this.n;
        boolean z = false;
        if (this.o) {
            int i = 0;
            for (b.v.e eVar : this.f349h) {
                if (!(eVar.f2615b instanceof k)) {
                    i++;
                }
            }
            if (i > 1) {
                z = true;
            }
        }
        bVar.f530a = z;
    }
}