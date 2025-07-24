package c.c.a.m.w.y;

import c.c.a.m.o;
import c.c.a.m.p;
import c.c.a.m.u.j;
import c.c.a.m.w.g;
import c.c.a.m.w.m;
import c.c.a.m.w.n;
import c.c.a.m.w.r;
import java.io.InputStream;
import java.util.Objects;
import java.util.Queue;

/* compiled from: HttpGlideUrlLoader.java */
/* loaded from: classes.dex */
public class a implements n<g, InputStream> {

    /* renamed from: a  reason: collision with root package name */
    public static final o<Integer> f3906a = o.a("com.bumptech.glide.load.model.stream.HttpGlideUrlLoader.Timeout", 2500);

    /* renamed from: b  reason: collision with root package name */
    public final m<g, g> f3907b;

    /* compiled from: HttpGlideUrlLoader.java */
    /* renamed from: c.c.a.m.w.y.a$a  reason: collision with other inner class name */
    /* loaded from: classes.dex */
    public static class C0077a implements c.c.a.m.w.o<g, InputStream> {

        /* renamed from: a  reason: collision with root package name */
        public final m<g, g> f3908a = new m<>(500);

        @Override // c.c.a.m.w.o
        public n<g, InputStream> b(r rVar) {
            return new a(this.f3908a);
        }
    }

    public a(m<g, g> mVar) {
        this.f3907b = mVar;
    }

    /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object] */
    @Override // c.c.a.m.w.n
    public /* bridge */ /* synthetic */ boolean a(g gVar) {
        return true;
    }

    /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object, int, int, c.c.a.m.p] */
    /* JADX DEBUG: Return type fixed from 'c.c.a.m.w.n$a' to match base method */
    @Override // c.c.a.m.w.n
    public n.a<InputStream> b(g gVar, int i, int i2, p pVar) {
        g gVar2 = gVar;
        m<g, g> mVar = this.f3907b;
        if (mVar != null) {
            m.b<g> a2 = m.b.a(gVar2, 0, 0);
            g a3 = mVar.f3858a.a(a2);
            Queue<m.b<?>> queue = m.b.f3859a;
            synchronized (queue) {
                queue.offer(a2);
            }
            g gVar3 = a3;
            if (gVar3 == null) {
                m<g, g> mVar2 = this.f3907b;
                Objects.requireNonNull(mVar2);
                mVar2.f3858a.d(m.b.a(gVar2, 0, 0), gVar2);
            } else {
                gVar2 = gVar3;
            }
        }
        return new n.a<>(gVar2, new j(gVar2, ((Integer) pVar.c(f3906a)).intValue()));
    }
}