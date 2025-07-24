package c.c.a.m.w;

import android.content.res.AssetFileDescriptor;
import android.content.res.Resources;
import android.net.Uri;
import android.os.ParcelFileDescriptor;
import android.util.Log;
import c.c.a.m.w.n;
import java.io.InputStream;

/* compiled from: ResourceLoader.java */
/* loaded from: classes.dex */
public class s<Data> implements n<Integer, Data> {

    /* renamed from: a  reason: collision with root package name */
    public final n<Uri, Data> f3888a;

    /* renamed from: b  reason: collision with root package name */
    public final Resources f3889b;

    /* compiled from: ResourceLoader.java */
    /* loaded from: classes.dex */
    public static final class a implements o<Integer, AssetFileDescriptor> {

        /* renamed from: a  reason: collision with root package name */
        public final Resources f3890a;

        public a(Resources resources) {
            this.f3890a = resources;
        }

        @Override // c.c.a.m.w.o
        public n<Integer, AssetFileDescriptor> b(r rVar) {
            return new s(this.f3890a, rVar.b(Uri.class, AssetFileDescriptor.class));
        }
    }

    /* compiled from: ResourceLoader.java */
    /* loaded from: classes.dex */
    public static class b implements o<Integer, ParcelFileDescriptor> {

        /* renamed from: a  reason: collision with root package name */
        public final Resources f3891a;

        public b(Resources resources) {
            this.f3891a = resources;
        }

        @Override // c.c.a.m.w.o
        public n<Integer, ParcelFileDescriptor> b(r rVar) {
            return new s(this.f3891a, rVar.b(Uri.class, ParcelFileDescriptor.class));
        }
    }

    /* compiled from: ResourceLoader.java */
    /* loaded from: classes.dex */
    public static class c implements o<Integer, InputStream> {

        /* renamed from: a  reason: collision with root package name */
        public final Resources f3892a;

        public c(Resources resources) {
            this.f3892a = resources;
        }

        @Override // c.c.a.m.w.o
        public n<Integer, InputStream> b(r rVar) {
            return new s(this.f3892a, rVar.b(Uri.class, InputStream.class));
        }
    }

    /* compiled from: ResourceLoader.java */
    /* loaded from: classes.dex */
    public static class d implements o<Integer, Uri> {

        /* renamed from: a  reason: collision with root package name */
        public final Resources f3893a;

        public d(Resources resources) {
            this.f3893a = resources;
        }

        @Override // c.c.a.m.w.o
        public n<Integer, Uri> b(r rVar) {
            return new s(this.f3893a, v.f3896a);
        }
    }

    public s(Resources resources, n<Uri, Data> nVar) {
        this.f3889b = resources;
        this.f3888a = nVar;
    }

    /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object] */
    @Override // c.c.a.m.w.n
    public /* bridge */ /* synthetic */ boolean a(Integer num) {
        return true;
    }

    /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object, int, int, c.c.a.m.p] */
    @Override // c.c.a.m.w.n
    public n.a b(Integer num, int i, int i2, c.c.a.m.p pVar) {
        Uri uri;
        Integer num2 = num;
        try {
            uri = Uri.parse("android.resource://" + this.f3889b.getResourcePackageName(num2.intValue()) + '/' + this.f3889b.getResourceTypeName(num2.intValue()) + '/' + this.f3889b.getResourceEntryName(num2.intValue()));
        } catch (Resources.NotFoundException e2) {
            if (Log.isLoggable("ResourceLoader", 5)) {
                Log.w("ResourceLoader", "Received invalid resource id: " + num2, e2);
            }
            uri = null;
        }
        if (uri == null) {
            return null;
        }
        return this.f3888a.b(uri, i, i2, pVar);
    }
}