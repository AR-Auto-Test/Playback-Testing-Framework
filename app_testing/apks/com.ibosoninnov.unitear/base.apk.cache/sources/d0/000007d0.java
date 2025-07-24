package c.c.a.m.w;

import c.c.a.m.u.d;
import c.c.a.m.w.n;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

/* compiled from: MultiModelLoader.java */
/* loaded from: classes.dex */
public class q<Model, Data> implements n<Model, Data> {

    /* renamed from: a  reason: collision with root package name */
    public final List<n<Model, Data>> f3870a;

    /* renamed from: b  reason: collision with root package name */
    public final b.j.i.d<List<Throwable>> f3871b;

    /* compiled from: MultiModelLoader.java */
    /* loaded from: classes.dex */
    public static class a<Data> implements c.c.a.m.u.d<Data>, d.a<Data> {

        /* renamed from: b  reason: collision with root package name */
        public final List<c.c.a.m.u.d<Data>> f3872b;

        /* renamed from: c  reason: collision with root package name */
        public final b.j.i.d<List<Throwable>> f3873c;

        /* renamed from: d  reason: collision with root package name */
        public int f3874d;

        /* renamed from: e  reason: collision with root package name */
        public c.c.a.f f3875e;

        /* renamed from: f  reason: collision with root package name */
        public d.a<? super Data> f3876f;

        /* renamed from: g  reason: collision with root package name */
        public List<Throwable> f3877g;

        /* renamed from: h  reason: collision with root package name */
        public boolean f3878h;

        public a(List<c.c.a.m.u.d<Data>> list, b.j.i.d<List<Throwable>> dVar) {
            this.f3873c = dVar;
            if (!list.isEmpty()) {
                this.f3872b = list;
                this.f3874d = 0;
                return;
            }
            throw new IllegalArgumentException("Must not be empty.");
        }

        @Override // c.c.a.m.u.d
        public Class<Data> a() {
            return this.f3872b.get(0).a();
        }

        @Override // c.c.a.m.u.d
        public void b() {
            List<Throwable> list = this.f3877g;
            if (list != null) {
                this.f3873c.a(list);
            }
            this.f3877g = null;
            for (c.c.a.m.u.d<Data> dVar : this.f3872b) {
                dVar.b();
            }
        }

        @Override // c.c.a.m.u.d.a
        public void c(Exception exc) {
            List<Throwable> list = this.f3877g;
            Objects.requireNonNull(list, "Argument must not be null");
            list.add(exc);
            g();
        }

        @Override // c.c.a.m.u.d
        public void cancel() {
            this.f3878h = true;
            for (c.c.a.m.u.d<Data> dVar : this.f3872b) {
                dVar.cancel();
            }
        }

        @Override // c.c.a.m.u.d
        public c.c.a.m.a d() {
            return this.f3872b.get(0).d();
        }

        @Override // c.c.a.m.u.d
        public void e(c.c.a.f fVar, d.a<? super Data> aVar) {
            this.f3875e = fVar;
            this.f3876f = aVar;
            this.f3877g = this.f3873c.b();
            this.f3872b.get(this.f3874d).e(fVar, this);
            if (this.f3878h) {
                cancel();
            }
        }

        @Override // c.c.a.m.u.d.a
        public void f(Data data) {
            if (data != null) {
                this.f3876f.f(data);
            } else {
                g();
            }
        }

        public final void g() {
            if (this.f3878h) {
                return;
            }
            if (this.f3874d < this.f3872b.size() - 1) {
                this.f3874d++;
                e(this.f3875e, this.f3876f);
                return;
            }
            Objects.requireNonNull(this.f3877g, "Argument must not be null");
            this.f3876f.c(new c.c.a.m.v.r("Fetch failed", new ArrayList(this.f3877g)));
        }
    }

    public q(List<n<Model, Data>> list, b.j.i.d<List<Throwable>> dVar) {
        this.f3870a = list;
        this.f3871b = dVar;
    }

    @Override // c.c.a.m.w.n
    public boolean a(Model model) {
        for (n<Model, Data> nVar : this.f3870a) {
            if (nVar.a(model)) {
                return true;
            }
        }
        return false;
    }

    @Override // c.c.a.m.w.n
    public n.a<Data> b(Model model, int i, int i2, c.c.a.m.p pVar) {
        n.a<Data> b2;
        int size = this.f3870a.size();
        ArrayList arrayList = new ArrayList(size);
        c.c.a.m.m mVar = null;
        for (int i3 = 0; i3 < size; i3++) {
            n<Model, Data> nVar = this.f3870a.get(i3);
            if (nVar.a(model) && (b2 = nVar.b(model, i, i2, pVar)) != null) {
                mVar = b2.f3863a;
                arrayList.add(b2.f3865c);
            }
        }
        if (arrayList.isEmpty() || mVar == null) {
            return null;
        }
        return new n.a<>(mVar, new a(arrayList, this.f3871b));
    }

    public String toString() {
        StringBuilder x = c.b.a.a.a.x("MultiModelLoader{modelLoaders=");
        x.append(Arrays.toString(this.f3870a.toArray()));
        x.append('}');
        return x.toString();
    }
}