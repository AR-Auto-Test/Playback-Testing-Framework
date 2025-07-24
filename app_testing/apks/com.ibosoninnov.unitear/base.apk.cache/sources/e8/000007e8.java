package c.c.a.m.w;

import android.net.Uri;
import c.c.a.m.w.n;
import java.io.InputStream;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.Set;

/* compiled from: UrlUriLoader.java */
/* loaded from: classes.dex */
public class x<Data> implements n<Uri, Data> {

    /* renamed from: a  reason: collision with root package name */
    public static final Set<String> f3904a = Collections.unmodifiableSet(new HashSet(Arrays.asList("http", "https")));

    /* renamed from: b  reason: collision with root package name */
    public final n<g, Data> f3905b;

    /* compiled from: UrlUriLoader.java */
    /* loaded from: classes.dex */
    public static class a implements o<Uri, InputStream> {
        @Override // c.c.a.m.w.o
        public n<Uri, InputStream> b(r rVar) {
            return new x(rVar.b(g.class, InputStream.class));
        }
    }

    public x(n<g, Data> nVar) {
        this.f3905b = nVar;
    }

    /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object] */
    @Override // c.c.a.m.w.n
    public boolean a(Uri uri) {
        return f3904a.contains(uri.getScheme());
    }

    /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object, int, int, c.c.a.m.p] */
    @Override // c.c.a.m.w.n
    public n.a b(Uri uri, int i, int i2, c.c.a.m.p pVar) {
        return this.f3905b.b(new g(uri.toString()), i, i2, pVar);
    }
}