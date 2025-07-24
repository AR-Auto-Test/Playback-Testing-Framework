package c.c.a.m.u;

import c.c.a.m.u.e;
import c.c.a.m.x.c.w;
import java.io.InputStream;

/* compiled from: InputStreamRewinder.java */
/* loaded from: classes.dex */
public final class k implements e<InputStream> {

    /* renamed from: a  reason: collision with root package name */
    public final w f3569a;

    /* compiled from: InputStreamRewinder.java */
    /* loaded from: classes.dex */
    public static final class a implements e.a<InputStream> {

        /* renamed from: a  reason: collision with root package name */
        public final c.c.a.m.v.c0.b f3570a;

        public a(c.c.a.m.v.c0.b bVar) {
            this.f3570a = bVar;
        }

        @Override // c.c.a.m.u.e.a
        public Class<InputStream> a() {
            return InputStream.class;
        }

        /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object] */
        /* JADX DEBUG: Return type fixed from 'c.c.a.m.u.e' to match base method */
        @Override // c.c.a.m.u.e.a
        public e<InputStream> b(InputStream inputStream) {
            return new k(inputStream, this.f3570a);
        }
    }

    public k(InputStream inputStream, c.c.a.m.v.c0.b bVar) {
        w wVar = new w(inputStream, bVar);
        this.f3569a = wVar;
        wVar.mark(5242880);
    }

    @Override // c.c.a.m.u.e
    public void b() {
        this.f3569a.release();
    }

    /* JADX DEBUG: Method merged with bridge method */
    @Override // c.c.a.m.u.e
    /* renamed from: c */
    public InputStream a() {
        this.f3569a.reset();
        return this.f3569a;
    }
}