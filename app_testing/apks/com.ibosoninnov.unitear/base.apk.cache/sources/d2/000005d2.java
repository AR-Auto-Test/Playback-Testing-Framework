package b.w.b;

import androidx.recyclerview.widget.RecyclerView;

/* compiled from: ViewInfoStore.java */
/* loaded from: classes.dex */
public class z {

    /* renamed from: a  reason: collision with root package name */
    public final b.f.h<RecyclerView.d0, a> f2814a = new b.f.h<>();

    /* renamed from: b  reason: collision with root package name */
    public final b.f.e<RecyclerView.d0> f2815b = new b.f.e<>(10);

    /* compiled from: ViewInfoStore.java */
    /* loaded from: classes.dex */
    public static class a {

        /* renamed from: a  reason: collision with root package name */
        public static b.j.i.d<a> f2816a = new b.j.i.e(20);

        /* renamed from: b  reason: collision with root package name */
        public int f2817b;

        /* renamed from: c  reason: collision with root package name */
        public RecyclerView.l.c f2818c;

        /* renamed from: d  reason: collision with root package name */
        public RecyclerView.l.c f2819d;

        public static a a() {
            a b2 = f2816a.b();
            return b2 == null ? new a() : b2;
        }

        public static void b(a aVar) {
            aVar.f2817b = 0;
            aVar.f2818c = null;
            aVar.f2819d = null;
            f2816a.a(aVar);
        }
    }

    /* compiled from: ViewInfoStore.java */
    /* loaded from: classes.dex */
    public interface b {
    }

    public void a(RecyclerView.d0 d0Var) {
        a orDefault = this.f2814a.getOrDefault(d0Var, null);
        if (orDefault == null) {
            orDefault = a.a();
            this.f2814a.put(d0Var, orDefault);
        }
        orDefault.f2817b |= 1;
    }

    public void b(RecyclerView.d0 d0Var, RecyclerView.l.c cVar) {
        a orDefault = this.f2814a.getOrDefault(d0Var, null);
        if (orDefault == null) {
            orDefault = a.a();
            this.f2814a.put(d0Var, orDefault);
        }
        orDefault.f2819d = cVar;
        orDefault.f2817b |= 8;
    }

    public void c(RecyclerView.d0 d0Var, RecyclerView.l.c cVar) {
        a orDefault = this.f2814a.getOrDefault(d0Var, null);
        if (orDefault == null) {
            orDefault = a.a();
            this.f2814a.put(d0Var, orDefault);
        }
        orDefault.f2818c = cVar;
        orDefault.f2817b |= 4;
    }

    public boolean d(RecyclerView.d0 d0Var) {
        a orDefault = this.f2814a.getOrDefault(d0Var, null);
        return (orDefault == null || (orDefault.f2817b & 1) == 0) ? false : true;
    }

    public final RecyclerView.l.c e(RecyclerView.d0 d0Var, int i) {
        a l;
        RecyclerView.l.c cVar;
        int e2 = this.f2814a.e(d0Var);
        if (e2 >= 0 && (l = this.f2814a.l(e2)) != null) {
            int i2 = l.f2817b;
            if ((i2 & i) != 0) {
                int i3 = (~i) & i2;
                l.f2817b = i3;
                if (i == 4) {
                    cVar = l.f2818c;
                } else if (i == 8) {
                    cVar = l.f2819d;
                } else {
                    throw new IllegalArgumentException("Must provide flag PRE or POST");
                }
                if ((i3 & 12) == 0) {
                    this.f2814a.j(e2);
                    a.b(l);
                }
                return cVar;
            }
        }
        return null;
    }

    public void f(RecyclerView.d0 d0Var) {
        a orDefault = this.f2814a.getOrDefault(d0Var, null);
        if (orDefault == null) {
            return;
        }
        orDefault.f2817b &= -2;
    }

    public void g(RecyclerView.d0 d0Var) {
        int h2 = this.f2815b.h() - 1;
        while (true) {
            if (h2 < 0) {
                break;
            } else if (d0Var == this.f2815b.i(h2)) {
                b.f.e<RecyclerView.d0> eVar = this.f2815b;
                Object[] objArr = eVar.f1752e;
                Object obj = objArr[h2];
                Object obj2 = b.f.e.f1749b;
                if (obj != obj2) {
                    objArr[h2] = obj2;
                    eVar.f1750c = true;
                }
            } else {
                h2--;
            }
        }
        a remove = this.f2814a.remove(d0Var);
        if (remove != null) {
            a.b(remove);
        }
    }
}