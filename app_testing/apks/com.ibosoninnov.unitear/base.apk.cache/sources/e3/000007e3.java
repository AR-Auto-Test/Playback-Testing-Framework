package c.c.a.m.w;

import android.content.ContentResolver;
import android.content.res.AssetFileDescriptor;
import android.net.Uri;
import android.os.ParcelFileDescriptor;
import c.c.a.m.w.n;
import com.google.firebase.analytics.FirebaseAnalytics;
import java.io.InputStream;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.Set;

/* compiled from: UriLoader.java */
/* loaded from: classes.dex */
public class w<Data> implements n<Uri, Data> {

    /* renamed from: a  reason: collision with root package name */
    public static final Set<String> f3899a = Collections.unmodifiableSet(new HashSet(Arrays.asList("file", "android.resource", FirebaseAnalytics.Param.CONTENT)));

    /* renamed from: b  reason: collision with root package name */
    public final c<Data> f3900b;

    /* compiled from: UriLoader.java */
    /* loaded from: classes.dex */
    public static final class a implements o<Uri, AssetFileDescriptor>, c<AssetFileDescriptor> {

        /* renamed from: a  reason: collision with root package name */
        public final ContentResolver f3901a;

        public a(ContentResolver contentResolver) {
            this.f3901a = contentResolver;
        }

        @Override // c.c.a.m.w.w.c
        public c.c.a.m.u.d<AssetFileDescriptor> a(Uri uri) {
            return new c.c.a.m.u.a(this.f3901a, uri);
        }

        @Override // c.c.a.m.w.o
        public n<Uri, AssetFileDescriptor> b(r rVar) {
            return new w(this);
        }
    }

    /* compiled from: UriLoader.java */
    /* loaded from: classes.dex */
    public static class b implements o<Uri, ParcelFileDescriptor>, c<ParcelFileDescriptor> {

        /* renamed from: a  reason: collision with root package name */
        public final ContentResolver f3902a;

        public b(ContentResolver contentResolver) {
            this.f3902a = contentResolver;
        }

        @Override // c.c.a.m.w.w.c
        public c.c.a.m.u.d<ParcelFileDescriptor> a(Uri uri) {
            return new c.c.a.m.u.i(this.f3902a, uri);
        }

        @Override // c.c.a.m.w.o
        public n<Uri, ParcelFileDescriptor> b(r rVar) {
            return new w(this);
        }
    }

    /* compiled from: UriLoader.java */
    /* loaded from: classes.dex */
    public interface c<Data> {
        c.c.a.m.u.d<Data> a(Uri uri);
    }

    /* compiled from: UriLoader.java */
    /* loaded from: classes.dex */
    public static class d implements o<Uri, InputStream>, c<InputStream> {

        /* renamed from: a  reason: collision with root package name */
        public final ContentResolver f3903a;

        public d(ContentResolver contentResolver) {
            this.f3903a = contentResolver;
        }

        @Override // c.c.a.m.w.w.c
        public c.c.a.m.u.d<InputStream> a(Uri uri) {
            return new c.c.a.m.u.n(this.f3903a, uri);
        }

        @Override // c.c.a.m.w.o
        public n<Uri, InputStream> b(r rVar) {
            return new w(this);
        }
    }

    public w(c<Data> cVar) {
        this.f3900b = cVar;
    }

    /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object] */
    @Override // c.c.a.m.w.n
    public boolean a(Uri uri) {
        return f3899a.contains(uri.getScheme());
    }

    /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object, int, int, c.c.a.m.p] */
    @Override // c.c.a.m.w.n
    public n.a b(Uri uri, int i, int i2, c.c.a.m.p pVar) {
        Uri uri2 = uri;
        return new n.a(new c.c.a.r.d(uri2), this.f3900b.a(uri2));
    }
}