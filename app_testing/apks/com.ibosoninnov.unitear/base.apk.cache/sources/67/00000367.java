package b.d.b.e1;

import android.util.Pair;
import android.util.Rational;
import android.util.Size;
import b.d.a.e.o1;
import b.d.a.e.v0;
import b.d.a.e.y1.p.h;
import b.d.a.e.y1.p.j;
import b.d.b.a1;
import b.d.b.d1.a0;
import b.d.b.d1.e1;
import b.d.b.d1.i0;
import b.d.b.d1.i1;
import b.d.b.d1.j1;
import b.d.b.d1.n0;
import b.d.b.d1.u;
import b.d.b.d1.v;
import b.d.b.d1.w;
import b.d.b.d1.x;
import b.d.b.d1.x0;
import b.d.b.d1.z;
import b.d.b.e0;
import b.d.b.f0;
import b.d.b.u0;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/* compiled from: CameraUseCaseAdapter.java */
/* loaded from: classes.dex */
public final class c implements e0 {

    /* renamed from: a  reason: collision with root package name */
    public a0 f1597a;

    /* renamed from: b  reason: collision with root package name */
    public final LinkedHashSet<a0> f1598b;

    /* renamed from: c  reason: collision with root package name */
    public final x f1599c;

    /* renamed from: d  reason: collision with root package name */
    public final j1 f1600d;

    /* renamed from: e  reason: collision with root package name */
    public final b f1601e;

    /* renamed from: f  reason: collision with root package name */
    public final List<a1> f1602f = new ArrayList();

    /* renamed from: g  reason: collision with root package name */
    public u f1603g = v.f1589a;

    /* renamed from: h  reason: collision with root package name */
    public final Object f1604h = new Object();
    public boolean i = true;
    public i0 j = null;

    /* compiled from: CameraUseCaseAdapter.java */
    /* loaded from: classes.dex */
    public static final class a extends Exception {
        public a(String str) {
            super(str);
        }
    }

    /* compiled from: CameraUseCaseAdapter.java */
    /* loaded from: classes.dex */
    public static final class b {

        /* renamed from: a  reason: collision with root package name */
        public final List<String> f1605a = new ArrayList();

        public b(LinkedHashSet<a0> linkedHashSet) {
            Iterator<a0> it = linkedHashSet.iterator();
            while (it.hasNext()) {
                this.f1605a.add(it.next().j().b());
            }
        }

        public boolean equals(Object obj) {
            if (obj instanceof b) {
                return this.f1605a.equals(((b) obj).f1605a);
            }
            return false;
        }

        public int hashCode() {
            return this.f1605a.hashCode() * 53;
        }
    }

    /* compiled from: CameraUseCaseAdapter.java */
    /* renamed from: b.d.b.e1.c$c  reason: collision with other inner class name */
    /* loaded from: classes.dex */
    public static class C0023c {

        /* renamed from: a  reason: collision with root package name */
        public i1<?> f1606a;

        /* renamed from: b  reason: collision with root package name */
        public i1<?> f1607b;

        public C0023c(i1<?> i1Var, i1<?> i1Var2) {
            this.f1606a = i1Var;
            this.f1607b = i1Var2;
        }
    }

    public c(LinkedHashSet<a0> linkedHashSet, x xVar, j1 j1Var) {
        this.f1597a = linkedHashSet.iterator().next();
        LinkedHashSet<a0> linkedHashSet2 = new LinkedHashSet<>(linkedHashSet);
        this.f1598b = linkedHashSet2;
        this.f1601e = new b(linkedHashSet2);
        this.f1599c = xVar;
        this.f1600d = j1Var;
    }

    @Override // b.d.b.e0
    public f0 a() {
        return this.f1597a.g();
    }

    @Override // b.d.b.e0
    public b.d.b.i0 b() {
        return this.f1597a.j();
    }

    public void c(Collection<a1> collection) {
        synchronized (this.f1604h) {
            ArrayList arrayList = new ArrayList();
            for (a1 a1Var : collection) {
                if (this.f1602f.contains(a1Var)) {
                    u0.a("CameraUseCaseAdapter", "Attempting to attach already attached UseCase", null);
                } else {
                    arrayList.add(a1Var);
                }
            }
            j1 j1Var = ((v.a) this.f1603g).q;
            j1 j1Var2 = this.f1600d;
            HashMap hashMap = new HashMap();
            Iterator it = arrayList.iterator();
            while (it.hasNext()) {
                a1 a1Var2 = (a1) it.next();
                hashMap.put(a1Var2, new C0023c(a1Var2.c(false, j1Var), a1Var2.c(true, j1Var2)));
            }
            try {
                Map<a1, Size> e2 = e(this.f1597a.j(), arrayList, this.f1602f, hashMap);
                synchronized (this.f1604h) {
                }
                Iterator it2 = arrayList.iterator();
                while (it2.hasNext()) {
                    a1 a1Var3 = (a1) it2.next();
                    C0023c c0023c = (C0023c) hashMap.get(a1Var3);
                    a1Var3.i(this.f1597a, c0023c.f1606a, c0023c.f1607b);
                    Size size = (Size) ((HashMap) e2).get(a1Var3);
                    Objects.requireNonNull(size);
                    a1Var3.f1385g = a1Var3.m(size);
                }
                this.f1602f.addAll(arrayList);
                if (this.i) {
                    this.f1597a.h(arrayList);
                }
                Iterator it3 = arrayList.iterator();
                while (it3.hasNext()) {
                    ((a1) it3.next()).h();
                }
            } catch (IllegalArgumentException e3) {
                throw new a(e3.getMessage());
            }
        }
    }

    public void d() {
        synchronized (this.f1604h) {
            if (!this.i) {
                this.f1597a.h(this.f1602f);
                synchronized (this.f1604h) {
                    if (this.j != null) {
                        this.f1597a.g().a(this.j);
                    }
                }
                for (a1 a1Var : this.f1602f) {
                    a1Var.h();
                }
                this.i = true;
            }
        }
    }

    /* JADX WARN: Code restructure failed: missing block: B:160:0x03de, code lost:
        if (b.d.a.e.o1.g(java.lang.Math.max(0, r4 - 16), r9, r14) == false) goto L139;
     */
    /* JADX WARN: Code restructure failed: missing block: B:61:0x01d6, code lost:
        if (b.d.a.e.o1.e(r0) < (r14.getHeight() * r14.getWidth())) goto L230;
     */
    /* JADX WARN: Removed duplicated region for block: B:132:0x030c  */
    /* JADX WARN: Removed duplicated region for block: B:135:0x031c  */
    /* JADX WARN: Removed duplicated region for block: B:139:0x032c  */
    /* JADX WARN: Removed duplicated region for block: B:170:0x03f9  */
    /* JADX WARN: Removed duplicated region for block: B:275:0x0409 A[SYNTHETIC] */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public final Map<a1, Size> e(z zVar, List<a1> list, List<a1> list2, Map<a1, C0023c> map) {
        List<Pair<Integer, Size[]>> list3;
        Size[] sizeArr;
        Size size;
        boolean z;
        Rational rational;
        Rational rational2;
        HashMap hashMap;
        HashMap hashMap2;
        ArrayList arrayList;
        Iterator it;
        Iterator it2;
        HashMap hashMap3;
        HashMap hashMap4;
        ArrayList arrayList2;
        boolean g2;
        ArrayList arrayList3 = new ArrayList();
        String b2 = zVar.b();
        HashMap hashMap5 = new HashMap();
        Iterator<a1> it3 = list2.iterator();
        while (true) {
            list3 = null;
            e1 e1Var = null;
            if (!it3.hasNext()) {
                break;
            }
            a1 next = it3.next();
            x xVar = this.f1599c;
            int l = next.f1384f.l();
            Size size2 = next.f1385g;
            o1 o1Var = ((v0) xVar).f1209a.get(b2);
            if (o1Var != null) {
                e1Var = o1Var.i(l, size2);
            }
            arrayList3.add(e1Var);
            hashMap5.put(next, next.f1385g);
        }
        if (list.isEmpty()) {
            return hashMap5;
        }
        HashMap hashMap6 = new HashMap();
        for (a1 a1Var : list) {
            C0023c c0023c = map.get(a1Var);
            hashMap6.put(a1Var.f(zVar, c0023c.f1606a, c0023c.f1607b), a1Var);
        }
        x xVar2 = this.f1599c;
        ArrayList arrayList4 = new ArrayList(hashMap6.keySet());
        v0 v0Var = (v0) xVar2;
        Objects.requireNonNull(v0Var);
        boolean z2 = true;
        b.j.b.d.e(!arrayList4.isEmpty(), "No new use cases to be bound.");
        ArrayList arrayList5 = new ArrayList(arrayList3);
        Iterator it4 = arrayList4.iterator();
        while (it4.hasNext()) {
            int l2 = ((i1) it4.next()).l();
            Size size3 = new Size(640, 480);
            o1 o1Var2 = v0Var.f1209a.get(b2);
            arrayList5.add(o1Var2 != null ? o1Var2.i(l2, size3) : null);
        }
        o1 o1Var3 = v0Var.f1209a.get(b2);
        if (o1Var3 != null) {
            if (o1Var3.a(arrayList5)) {
                HashMap hashMap7 = new HashMap();
                ArrayList arrayList6 = new ArrayList();
                ArrayList arrayList7 = new ArrayList();
                Iterator it5 = arrayList4.iterator();
                while (it5.hasNext()) {
                    int r = ((i1) it5.next()).r(0);
                    if (!arrayList7.contains(Integer.valueOf(r))) {
                        arrayList7.add(Integer.valueOf(r));
                    }
                }
                Collections.sort(arrayList7);
                Collections.reverse(arrayList7);
                Iterator it6 = arrayList7.iterator();
                while (it6.hasNext()) {
                    int intValue = ((Integer) it6.next()).intValue();
                    Iterator it7 = arrayList4.iterator();
                    while (it7.hasNext()) {
                        i1 i1Var = (i1) it7.next();
                        if (intValue == i1Var.r(0)) {
                            arrayList6.add(Integer.valueOf(arrayList4.indexOf(i1Var)));
                        }
                    }
                }
                ArrayList arrayList8 = new ArrayList();
                Iterator it8 = arrayList6.iterator();
                while (it8.hasNext()) {
                    i1 i1Var2 = (i1) arrayList4.get(((Integer) it8.next()).intValue());
                    int l3 = i1Var2.l();
                    n0 n0Var = (n0) i1Var2;
                    List<Pair<Integer, Size[]>> j = n0Var.j(list3);
                    if (j != null) {
                        for (Pair<Integer, Size[]> pair : j) {
                            if (((Integer) pair.first).intValue() == l3) {
                                sizeArr = (Size[]) pair.second;
                                break;
                            }
                        }
                    }
                    sizeArr = null;
                    if (sizeArr != null) {
                        sizeArr = o1Var3.b(sizeArr, l3);
                        Arrays.sort(sizeArr, new o1.b(z2));
                    }
                    if (sizeArr == null) {
                        sizeArr = o1Var3.d(l3);
                    }
                    ArrayList arrayList9 = new ArrayList();
                    Size i = n0Var.i(null);
                    Size size4 = (Size) Collections.max(Arrays.asList(o1Var3.d(l3)), new o1.b());
                    Size size5 = i != null ? size4 : size4;
                    i = size5;
                    Arrays.sort(sizeArr, new o1.b(true));
                    Size f2 = o1Var3.f(n0Var);
                    Size size6 = o1.f1123b;
                    Iterator it9 = it8;
                    int e2 = o1.e(size6);
                    if (o1.e(i) < e2) {
                        size = o1.f1124c;
                    } else {
                        size = (f2 == null || f2.getWidth() * f2.getHeight() >= e2) ? size6 : f2;
                    }
                    int length = sizeArr.length;
                    HashMap hashMap8 = hashMap5;
                    int i2 = 0;
                    while (i2 < length) {
                        int i3 = length;
                        Size size7 = sizeArr[i2];
                        Size[] sizeArr2 = sizeArr;
                        Size size8 = i;
                        if (o1.e(size7) <= i.getHeight() * i.getWidth()) {
                            if (size7.getHeight() * size7.getWidth() >= o1.e(size) && !arrayList9.contains(size7)) {
                                arrayList9.add(size7);
                            }
                        }
                        i2++;
                        sizeArr = sizeArr2;
                        i = size8;
                        length = i3;
                    }
                    if (!arrayList9.isEmpty()) {
                        b.d.a.e.y1.e eVar = o1Var3.p;
                        if (((j) b.d.a.e.y1.p.d.a(j.class)) == null || !(n0Var instanceof x0)) {
                            z = (((h) b.d.a.e.y1.p.d.a(h.class)) == null && ((b.d.a.e.y1.p.b) b.b.a.h(eVar).a(b.d.a.e.y1.p.b.class)) == null) ? true : true;
                        } else {
                            z = true;
                        }
                        if (!z) {
                            rational = o1Var3.s ? o1.f1129h : o1.i;
                        } else if (z) {
                            rational = o1Var3.s ? o1.j : o1.k;
                        } else {
                            if (!z) {
                                if (z) {
                                    Size f3 = o1Var3.f(n0Var);
                                    if (n0Var.q()) {
                                        int s = n0Var.s();
                                        if (s == 0) {
                                            rational = o1Var3.s ? o1.f1129h : o1.i;
                                        } else if (s != 1) {
                                            u0.b("SupportedSurfaceCombination", "Undefined target aspect ratio: " + s, null);
                                        } else {
                                            rational = o1Var3.s ? o1.j : o1.k;
                                        }
                                    } else if (f3 != null) {
                                        rational2 = new Rational(f3.getWidth(), f3.getHeight());
                                    }
                                }
                                rational2 = null;
                            } else {
                                Size c2 = o1Var3.c(256);
                                rational2 = new Rational(c2.getWidth(), c2.getHeight());
                            }
                            Rational rational3 = null;
                            if (f2 == null) {
                                f2 = n0Var.n(null);
                            }
                            ArrayList arrayList10 = new ArrayList();
                            new HashMap();
                            if (rational2 != null) {
                                arrayList10.addAll(arrayList9);
                                if (f2 != null) {
                                    o1Var3.h(arrayList10, f2);
                                }
                                hashMap = hashMap7;
                                hashMap2 = hashMap6;
                                arrayList = arrayList4;
                            } else {
                                HashMap hashMap9 = new HashMap();
                                hashMap9.put(o1.f1129h, new ArrayList());
                                hashMap9.put(o1.j, new ArrayList());
                                Iterator it10 = arrayList9.iterator();
                                while (it10.hasNext()) {
                                    Size size9 = (Size) it10.next();
                                    Iterator it11 = hashMap9.keySet().iterator();
                                    while (it11.hasNext()) {
                                        Rational rational4 = (Rational) it11.next();
                                        if (rational4 == null) {
                                            hashMap4 = hashMap7;
                                            hashMap3 = hashMap6;
                                            arrayList2 = arrayList4;
                                            it = it10;
                                            it2 = it11;
                                        } else {
                                            it = it10;
                                            it2 = it11;
                                            hashMap3 = hashMap6;
                                            if (rational4.equals(new Rational(size9.getWidth(), size9.getHeight()))) {
                                                hashMap4 = hashMap7;
                                                arrayList2 = arrayList4;
                                            } else if (size9.getHeight() * size9.getWidth() >= o1.e(o1.f1123b)) {
                                                int width = size9.getWidth();
                                                int height = size9.getHeight();
                                                hashMap4 = hashMap7;
                                                arrayList2 = arrayList4;
                                                Rational rational5 = new Rational(rational4.getDenominator(), rational4.getNumerator());
                                                int i4 = width % 16;
                                                if (i4 != 0 || height % 16 != 0) {
                                                    if (i4 == 0) {
                                                        g2 = o1.g(height, width, rational4);
                                                    } else if (height % 16 == 0) {
                                                        g2 = o1.g(width, height, rational5);
                                                    }
                                                    if (g2) {
                                                        List list4 = (List) hashMap9.get(rational4);
                                                        if (!list4.contains(size9)) {
                                                            list4.add(size9);
                                                        }
                                                        rational3 = rational4;
                                                    }
                                                    it10 = it;
                                                    it11 = it2;
                                                    hashMap7 = hashMap4;
                                                    hashMap6 = hashMap3;
                                                    arrayList4 = arrayList2;
                                                } else if (!o1.g(Math.max(0, height - 16), width, rational4)) {
                                                }
                                            } else {
                                                hashMap4 = hashMap7;
                                                arrayList2 = arrayList4;
                                            }
                                            g2 = true;
                                            if (g2) {
                                            }
                                            it10 = it;
                                            it11 = it2;
                                            hashMap7 = hashMap4;
                                            hashMap6 = hashMap3;
                                            arrayList4 = arrayList2;
                                        }
                                        g2 = false;
                                        if (g2) {
                                        }
                                        it10 = it;
                                        it11 = it2;
                                        hashMap7 = hashMap4;
                                        hashMap6 = hashMap3;
                                        arrayList4 = arrayList2;
                                    }
                                    HashMap hashMap10 = hashMap7;
                                    HashMap hashMap11 = hashMap6;
                                    ArrayList arrayList11 = arrayList4;
                                    Iterator it12 = it10;
                                    if (rational3 == null) {
                                        hashMap9.put(new Rational(size9.getWidth(), size9.getHeight()), new ArrayList(Collections.singleton(size9)));
                                    }
                                    rational3 = null;
                                    it10 = it12;
                                    hashMap7 = hashMap10;
                                    hashMap6 = hashMap11;
                                    arrayList4 = arrayList11;
                                }
                                hashMap = hashMap7;
                                hashMap2 = hashMap6;
                                arrayList = arrayList4;
                                if (f2 != null) {
                                    for (Rational rational6 : hashMap9.keySet()) {
                                        o1Var3.h((List) hashMap9.get(rational6), f2);
                                    }
                                }
                                ArrayList arrayList12 = new ArrayList(hashMap9.keySet());
                                Collections.sort(arrayList12, new o1.a(rational2));
                                Iterator it13 = arrayList12.iterator();
                                while (it13.hasNext()) {
                                    for (Size size10 : (List) hashMap9.get((Rational) it13.next())) {
                                        if (!arrayList10.contains(size10)) {
                                            arrayList10.add(size10);
                                        }
                                    }
                                }
                            }
                            arrayList8.add(arrayList10);
                            list3 = null;
                            z2 = true;
                            it8 = it9;
                            hashMap7 = hashMap;
                            hashMap6 = hashMap2;
                            hashMap5 = hashMap8;
                            arrayList4 = arrayList;
                        }
                        rational2 = rational;
                        Rational rational32 = null;
                        if (f2 == null) {
                        }
                        ArrayList arrayList102 = new ArrayList();
                        new HashMap();
                        if (rational2 != null) {
                        }
                        arrayList8.add(arrayList102);
                        list3 = null;
                        z2 = true;
                        it8 = it9;
                        hashMap7 = hashMap;
                        hashMap6 = hashMap2;
                        hashMap5 = hashMap8;
                        arrayList4 = arrayList;
                    } else {
                        throw new IllegalArgumentException(c.b.a.a.a.j("Can not get supported output size under supported maximum for the format: ", l3));
                    }
                }
                HashMap hashMap12 = hashMap7;
                HashMap hashMap13 = hashMap5;
                HashMap hashMap14 = hashMap6;
                ArrayList arrayList13 = arrayList4;
                Iterator it14 = arrayList8.iterator();
                int i5 = 1;
                while (it14.hasNext()) {
                    i5 *= ((List) it14.next()).size();
                }
                if (i5 != 0) {
                    ArrayList arrayList14 = new ArrayList();
                    for (int i6 = 0; i6 < i5; i6++) {
                        arrayList14.add(new ArrayList());
                    }
                    int size11 = i5 / ((List) arrayList8.get(0)).size();
                    int i7 = i5;
                    for (int i8 = 0; i8 < arrayList8.size(); i8++) {
                        List list5 = (List) arrayList8.get(i8);
                        for (int i9 = 0; i9 < i5; i9++) {
                            ((List) arrayList14.get(i9)).add((Size) list5.get((i9 % i7) / size11));
                        }
                        if (i8 < arrayList8.size() - 1) {
                            i7 = size11;
                            size11 /= ((List) arrayList8.get(i8 + 1)).size();
                        }
                    }
                    Iterator it15 = arrayList14.iterator();
                    while (true) {
                        if (!it15.hasNext()) {
                            break;
                        }
                        List list6 = (List) it15.next();
                        ArrayList arrayList15 = new ArrayList(arrayList3);
                        for (int i10 = 0; i10 < list6.size(); i10++) {
                            arrayList15.add(o1Var3.i(((i1) arrayList13.get(((Integer) arrayList6.get(i10)).intValue())).l(), (Size) list6.get(i10)));
                        }
                        ArrayList arrayList16 = arrayList13;
                        if (o1Var3.a(arrayList15)) {
                            Iterator it16 = arrayList16.iterator();
                            while (it16.hasNext()) {
                                i1 i1Var3 = (i1) it16.next();
                                hashMap12.put(i1Var3, (Size) list6.get(arrayList6.indexOf(Integer.valueOf(arrayList16.indexOf(i1Var3)))));
                            }
                        } else {
                            arrayList13 = arrayList16;
                        }
                    }
                    for (Map.Entry entry : hashMap14.entrySet()) {
                        hashMap13.put((a1) entry.getValue(), (Size) hashMap12.get(entry.getKey()));
                    }
                    return hashMap13;
                }
                throw new IllegalArgumentException("Failed to find supported resolutions.");
            }
            throw new IllegalArgumentException("No supported surface combination is found for camera device - Id : " + b2 + ".  May be attempting to bind too many use cases. Existing surfaces: " + arrayList3 + " New configs: " + arrayList4);
        }
        throw new IllegalArgumentException(c.b.a.a.a.q("No such camera id in supported combination list: ", b2));
    }

    public void f() {
        synchronized (this.f1604h) {
            if (this.i) {
                synchronized (this.f1604h) {
                    w g2 = this.f1597a.g();
                    this.j = g2.c();
                    g2.d();
                }
                this.f1597a.i(new ArrayList(this.f1602f));
                this.i = false;
            }
        }
    }

    public List<a1> k() {
        ArrayList arrayList;
        synchronized (this.f1604h) {
            arrayList = new ArrayList(this.f1602f);
        }
        return arrayList;
    }

    public void l(Collection<a1> collection) {
        synchronized (this.f1604h) {
            this.f1597a.i(collection);
            for (a1 a1Var : collection) {
                if (this.f1602f.contains(a1Var)) {
                    a1Var.j(this.f1597a);
                } else {
                    u0.b("CameraUseCaseAdapter", "Attempting to detach non-attached UseCase: " + a1Var, null);
                }
            }
            this.f1602f.removeAll(collection);
        }
    }
}