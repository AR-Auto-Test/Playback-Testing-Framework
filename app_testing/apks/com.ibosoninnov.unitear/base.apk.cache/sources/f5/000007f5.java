package c.c.a.m.w.y;

import c.c.a.m.p;
import c.c.a.m.w.g;
import c.c.a.m.w.n;
import c.c.a.m.w.o;
import c.c.a.m.w.r;
import java.io.InputStream;
import java.net.URL;

/* compiled from: UrlLoader.java */
/* loaded from: classes.dex */
public class e implements n<URL, InputStream> {

    /* renamed from: a  reason: collision with root package name */
    public final n<g, InputStream> f3926a;

    /* compiled from: UrlLoader.java */
    /* loaded from: classes.dex */
    public static class a implements o<URL, InputStream> {
        @Override // c.c.a.m.w.o
        public n<URL, InputStream> b(r rVar) {
            return new e(rVar.b(g.class, InputStream.class));
        }
    }

    public e(n<g, InputStream> nVar) {
        this.f3926a = nVar;
    }

    /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object] */
    @Override // c.c.a.m.w.n
    public /* bridge */ /* synthetic */ boolean a(URL url) {
        return true;
    }

    /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object, int, int, c.c.a.m.p] */
    /* JADX DEBUG: Return type fixed from 'c.c.a.m.w.n$a' to match base method */
    @Override // c.c.a.m.w.n
    public n.a<InputStream> b(URL url, int i, int i2, p pVar) {
        return this.f3926a.b(new g(url), i, i2, pVar);
    }
}