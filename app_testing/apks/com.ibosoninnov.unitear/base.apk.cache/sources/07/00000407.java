package b.h.b.i.l;

import java.lang.ref.WeakReference;
import java.util.ArrayList;
import java.util.Iterator;

/* compiled from: WidgetGroup.java */
/* loaded from: classes.dex */
public class n {

    /* renamed from: a  reason: collision with root package name */
    public static int f1922a;

    /* renamed from: c  reason: collision with root package name */
    public int f1924c;

    /* renamed from: d  reason: collision with root package name */
    public int f1925d;

    /* renamed from: b  reason: collision with root package name */
    public ArrayList<b.h.b.i.d> f1923b = new ArrayList<>();

    /* renamed from: e  reason: collision with root package name */
    public ArrayList<a> f1926e = null;

    /* renamed from: f  reason: collision with root package name */
    public int f1927f = -1;

    /* compiled from: WidgetGroup.java */
    /* loaded from: classes.dex */
    public class a {
        public a(n nVar, b.h.b.i.d dVar, b.h.b.d dVar2, int i) {
            new WeakReference(dVar);
            dVar2.o(dVar.D);
            dVar2.o(dVar.E);
            dVar2.o(dVar.F);
            dVar2.o(dVar.G);
            dVar2.o(dVar.H);
        }
    }

    public n(int i) {
        this.f1924c = -1;
        this.f1925d = 0;
        int i2 = f1922a;
        f1922a = i2 + 1;
        this.f1924c = i2;
        this.f1925d = i;
    }

    public boolean a(b.h.b.i.d dVar) {
        if (this.f1923b.contains(dVar)) {
            return false;
        }
        this.f1923b.add(dVar);
        return true;
    }

    public void b(ArrayList<n> arrayList) {
        int size = this.f1923b.size();
        if (this.f1927f != -1 && size > 0) {
            for (int i = 0; i < arrayList.size(); i++) {
                n nVar = arrayList.get(i);
                if (this.f1927f == nVar.f1924c) {
                    d(this.f1925d, nVar);
                }
            }
        }
        if (size == 0) {
            arrayList.remove(this);
        }
    }

    public int c(b.h.b.d dVar, int i) {
        int o;
        int o2;
        if (this.f1923b.size() == 0) {
            return 0;
        }
        ArrayList<b.h.b.i.d> arrayList = this.f1923b;
        b.h.b.i.e eVar = (b.h.b.i.e) arrayList.get(0).P;
        dVar.u();
        eVar.d(dVar, false);
        for (int i2 = 0; i2 < arrayList.size(); i2++) {
            arrayList.get(i2).d(dVar, false);
        }
        if (i == 0 && eVar.t0 > 0) {
            b.e.a.a(eVar, dVar, arrayList, 0);
        }
        if (i == 1 && eVar.u0 > 0) {
            b.e.a.a(eVar, dVar, arrayList, 1);
        }
        try {
            dVar.q();
        } catch (Exception e2) {
            e2.printStackTrace();
        }
        this.f1926e = new ArrayList<>();
        for (int i3 = 0; i3 < arrayList.size(); i3++) {
            this.f1926e.add(new a(this, arrayList.get(i3), dVar, i));
        }
        if (i == 0) {
            o = dVar.o(eVar.D);
            o2 = dVar.o(eVar.F);
            dVar.u();
        } else {
            o = dVar.o(eVar.E);
            o2 = dVar.o(eVar.G);
            dVar.u();
        }
        return o2 - o;
    }

    public void d(int i, n nVar) {
        Iterator<b.h.b.i.d> it = this.f1923b.iterator();
        while (it.hasNext()) {
            b.h.b.i.d next = it.next();
            nVar.a(next);
            if (i == 0) {
                next.j0 = nVar.f1924c;
            } else {
                next.k0 = nVar.f1924c;
            }
        }
        this.f1927f = nVar.f1924c;
    }

    public String toString() {
        StringBuilder sb = new StringBuilder();
        int i = this.f1925d;
        sb.append(i == 0 ? "Horizontal" : i == 1 ? "Vertical" : i == 2 ? "Both" : "Unknown");
        sb.append(" [");
        String s = c.b.a.a.a.s(sb, this.f1924c, "] <");
        Iterator<b.h.b.i.d> it = this.f1923b.iterator();
        while (it.hasNext()) {
            StringBuilder A = c.b.a.a.a.A(s, " ");
            A.append(it.next().d0);
            s = A.toString();
        }
        return c.b.a.a.a.q(s, " >");
    }
}