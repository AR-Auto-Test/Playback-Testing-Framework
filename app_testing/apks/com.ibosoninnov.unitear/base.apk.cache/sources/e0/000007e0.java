package c.c.a.m.w;

import c.c.a.m.u.d;
import c.c.a.m.w.n;

/* compiled from: UnitModelLoader.java */
/* loaded from: classes.dex */
public class v<Model> implements n<Model, Model> {

    /* renamed from: a  reason: collision with root package name */
    public static final v<?> f3896a = new v<>();

    /* compiled from: UnitModelLoader.java */
    /* loaded from: classes.dex */
    public static class a<Model> implements o<Model, Model> {

        /* renamed from: a  reason: collision with root package name */
        public static final a<?> f3897a = new a<>();

        @Override // c.c.a.m.w.o
        public n<Model, Model> b(r rVar) {
            return v.f3896a;
        }
    }

    /* compiled from: UnitModelLoader.java */
    /* loaded from: classes.dex */
    public static class b<Model> implements c.c.a.m.u.d<Model> {

        /* renamed from: b  reason: collision with root package name */
        public final Model f3898b;

        public b(Model model) {
            this.f3898b = model;
        }

        @Override // c.c.a.m.u.d
        public Class<Model> a() {
            return (Class<Model>) this.f3898b.getClass();
        }

        @Override // c.c.a.m.u.d
        public void b() {
        }

        @Override // c.c.a.m.u.d
        public void cancel() {
        }

        @Override // c.c.a.m.u.d
        public c.c.a.m.a d() {
            return c.c.a.m.a.LOCAL;
        }

        /* JADX DEBUG: Type inference failed for r1v1. Raw type applied. Possible types: Model, ? super Model */
        @Override // c.c.a.m.u.d
        public void e(c.c.a.f fVar, d.a<? super Model> aVar) {
            aVar.f((Model) this.f3898b);
        }
    }

    @Override // c.c.a.m.w.n
    public boolean a(Model model) {
        return true;
    }

    @Override // c.c.a.m.w.n
    public n.a<Model> b(Model model, int i, int i2, c.c.a.m.p pVar) {
        return new n.a<>(new c.c.a.r.d(model), new b(model));
    }
}