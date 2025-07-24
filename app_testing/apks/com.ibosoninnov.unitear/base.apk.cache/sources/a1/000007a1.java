package c.c.a.m.w;

import android.content.res.AssetManager;
import android.net.Uri;
import android.os.ParcelFileDescriptor;
import c.c.a.m.w.n;
import java.io.InputStream;

/* compiled from: AssetUriLoader.java */
/* loaded from: classes.dex */
public class a<Data> implements n<Uri, Data> {

    /* renamed from: a  reason: collision with root package name */
    public final AssetManager f3821a;

    /* renamed from: b  reason: collision with root package name */
    public final InterfaceC0073a<Data> f3822b;

    /* compiled from: AssetUriLoader.java */
    /* renamed from: c.c.a.m.w.a$a  reason: collision with other inner class name */
    /* loaded from: classes.dex */
    public interface InterfaceC0073a<Data> {
        c.c.a.m.u.d<Data> a(AssetManager assetManager, String str);
    }

    /* compiled from: AssetUriLoader.java */
    /* loaded from: classes.dex */
    public static class b implements o<Uri, ParcelFileDescriptor>, InterfaceC0073a<ParcelFileDescriptor> {

        /* renamed from: a  reason: collision with root package name */
        public final AssetManager f3823a;

        public b(AssetManager assetManager) {
            this.f3823a = assetManager;
        }

        @Override // c.c.a.m.w.a.InterfaceC0073a
        public c.c.a.m.u.d<ParcelFileDescriptor> a(AssetManager assetManager, String str) {
            return new c.c.a.m.u.h(assetManager, str);
        }

        @Override // c.c.a.m.w.o
        public n<Uri, ParcelFileDescriptor> b(r rVar) {
            return new a(this.f3823a, this);
        }
    }

    /* compiled from: AssetUriLoader.java */
    /* loaded from: classes.dex */
    public static class c implements o<Uri, InputStream>, InterfaceC0073a<InputStream> {

        /* renamed from: a  reason: collision with root package name */
        public final AssetManager f3824a;

        public c(AssetManager assetManager) {
            this.f3824a = assetManager;
        }

        @Override // c.c.a.m.w.a.InterfaceC0073a
        public c.c.a.m.u.d<InputStream> a(AssetManager assetManager, String str) {
            return new c.c.a.m.u.m(assetManager, str);
        }

        @Override // c.c.a.m.w.o
        public n<Uri, InputStream> b(r rVar) {
            return new a(this.f3824a, this);
        }
    }

    public a(AssetManager assetManager, InterfaceC0073a<Data> interfaceC0073a) {
        this.f3821a = assetManager;
        this.f3822b = interfaceC0073a;
    }

    /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object] */
    @Override // c.c.a.m.w.n
    public boolean a(Uri uri) {
        Uri uri2 = uri;
        return "file".equals(uri2.getScheme()) && !uri2.getPathSegments().isEmpty() && "android_asset".equals(uri2.getPathSegments().get(0));
    }

    /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object, int, int, c.c.a.m.p] */
    @Override // c.c.a.m.w.n
    public n.a b(Uri uri, int i, int i2, c.c.a.m.p pVar) {
        Uri uri2 = uri;
        return new n.a(new c.c.a.r.d(uri2), this.f3822b.a(this.f3821a, uri2.toString().substring(22)));
    }
}