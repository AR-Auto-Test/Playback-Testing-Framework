package b.j.c.b;

import android.graphics.Typeface;
import android.os.Handler;
import android.os.Looper;

/* compiled from: ResourcesCompat.java */
/* loaded from: classes.dex */
public abstract class e {

    /* compiled from: ResourcesCompat.java */
    /* loaded from: classes.dex */
    public class a implements Runnable {

        /* renamed from: b  reason: collision with root package name */
        public final /* synthetic */ Typeface f2087b;

        public a(Typeface typeface) {
            this.f2087b = typeface;
        }

        @Override // java.lang.Runnable
        public void run() {
            e.this.onFontRetrieved(this.f2087b);
        }
    }

    /* compiled from: ResourcesCompat.java */
    /* loaded from: classes.dex */
    public class b implements Runnable {

        /* renamed from: b  reason: collision with root package name */
        public final /* synthetic */ int f2089b;

        public b(int i) {
            this.f2089b = i;
        }

        @Override // java.lang.Runnable
        public void run() {
            e.this.onFontRetrievalFailed(this.f2089b);
        }
    }

    public static Handler getHandler(Handler handler) {
        return handler == null ? new Handler(Looper.getMainLooper()) : handler;
    }

    public final void callbackFailAsync(int i, Handler handler) {
        getHandler(handler).post(new b(i));
    }

    public final void callbackSuccessAsync(Typeface typeface, Handler handler) {
        getHandler(handler).post(new a(typeface));
    }

    public abstract void onFontRetrievalFailed(int i);

    public abstract void onFontRetrieved(Typeface typeface);
}