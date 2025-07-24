package b.j.j.y;

import android.content.ClipDescription;
import android.net.Uri;
import android.os.Build;
import android.view.inputmethod.InputContentInfo;

/* compiled from: InputContentInfoCompat.java */
/* loaded from: classes.dex */
public final class e {

    /* renamed from: a  reason: collision with root package name */
    public final c f2277a;

    /* compiled from: InputContentInfoCompat.java */
    /* loaded from: classes.dex */
    public static final class b implements c {

        /* renamed from: a  reason: collision with root package name */
        public final Uri f2279a;

        /* renamed from: b  reason: collision with root package name */
        public final ClipDescription f2280b;

        public b(Uri uri, ClipDescription clipDescription, Uri uri2) {
            this.f2279a = uri;
            this.f2280b = clipDescription;
        }

        @Override // b.j.j.y.e.c
        public Uri a() {
            return this.f2279a;
        }

        @Override // b.j.j.y.e.c
        public void b() {
        }

        @Override // b.j.j.y.e.c
        public ClipDescription c() {
            return this.f2280b;
        }

        @Override // b.j.j.y.e.c
        public void d() {
        }
    }

    /* compiled from: InputContentInfoCompat.java */
    /* loaded from: classes.dex */
    public interface c {
        Uri a();

        void b();

        ClipDescription c();

        void d();
    }

    public e(Uri uri, ClipDescription clipDescription, Uri uri2) {
        if (Build.VERSION.SDK_INT >= 25) {
            this.f2277a = new a(uri, clipDescription, uri2);
        } else {
            this.f2277a = new b(uri, clipDescription, uri2);
        }
    }

    /* compiled from: InputContentInfoCompat.java */
    /* loaded from: classes.dex */
    public static final class a implements c {

        /* renamed from: a  reason: collision with root package name */
        public final InputContentInfo f2278a;

        public a(Object obj) {
            this.f2278a = (InputContentInfo) obj;
        }

        @Override // b.j.j.y.e.c
        public Uri a() {
            return this.f2278a.getContentUri();
        }

        @Override // b.j.j.y.e.c
        public void b() {
            this.f2278a.requestPermission();
        }

        @Override // b.j.j.y.e.c
        public ClipDescription c() {
            return this.f2278a.getDescription();
        }

        @Override // b.j.j.y.e.c
        public void d() {
            this.f2278a.releasePermission();
        }

        public a(Uri uri, ClipDescription clipDescription, Uri uri2) {
            this.f2278a = new InputContentInfo(uri, clipDescription, uri2);
        }
    }

    public e(c cVar) {
        this.f2277a = cVar;
    }
}