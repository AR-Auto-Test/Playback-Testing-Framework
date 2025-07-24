package h.a.a;

import android.content.ContentResolver;
import android.content.res.AssetFileDescriptor;
import android.content.res.ColorStateList;
import android.content.res.Resources;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.ColorFilter;
import android.graphics.Paint;
import android.graphics.PorterDuff;
import android.graphics.PorterDuffColorFilter;
import android.graphics.Rect;
import android.graphics.drawable.Animatable;
import android.graphics.drawable.Drawable;
import android.net.Uri;
import android.os.SystemClock;
import android.util.TypedValue;
import android.widget.MediaController;
import h.a.a.e;
import java.io.IOException;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.ScheduledThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import pl.droidsonroids.gif.GifInfoHandle;

/* compiled from: GifDrawable.java */
/* loaded from: classes2.dex */
public class c extends Drawable implements Animatable, MediaController.MediaPlayerControl {

    /* renamed from: b  reason: collision with root package name */
    public final ScheduledThreadPoolExecutor f6226b;

    /* renamed from: c  reason: collision with root package name */
    public volatile boolean f6227c;

    /* renamed from: d  reason: collision with root package name */
    public long f6228d;

    /* renamed from: e  reason: collision with root package name */
    public final Rect f6229e;

    /* renamed from: f  reason: collision with root package name */
    public final Paint f6230f;

    /* renamed from: g  reason: collision with root package name */
    public final Bitmap f6231g;

    /* renamed from: h  reason: collision with root package name */
    public final GifInfoHandle f6232h;
    public final ConcurrentLinkedQueue<h.a.a.a> i;
    public ColorStateList j;
    public PorterDuffColorFilter k;
    public PorterDuff.Mode l;
    public final boolean m;
    public final h n;
    public final j o;
    public final Rect p;
    public ScheduledFuture<?> q;
    public int r;
    public int s;

    /* compiled from: GifDrawable.java */
    /* loaded from: classes2.dex */
    public class a extends k {

        /* renamed from: c  reason: collision with root package name */
        public final /* synthetic */ int f6233c;

        /* JADX WARN: 'super' call moved to the top of the method (can break code semantics) */
        public a(c cVar, int i) {
            super(cVar);
            this.f6233c = i;
        }

        @Override // h.a.a.k
        public void a() {
            c cVar = c.this;
            GifInfoHandle gifInfoHandle = cVar.f6232h;
            int i = this.f6233c;
            Bitmap bitmap = cVar.f6231g;
            synchronized (gifInfoHandle) {
                GifInfoHandle.seekToTime(gifInfoHandle.f6268b, i, bitmap);
            }
            this.f6252b.n.sendEmptyMessageAtTime(-1, 0L);
        }
    }

    /* JADX WARN: Illegal instructions before constructor call */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public c(ContentResolver contentResolver, Uri uri) {
        this(r3, null, null, true);
        GifInfoHandle gifInfoHandle;
        int i = GifInfoHandle.f6267a;
        if ("file".equals(uri.getScheme())) {
            gifInfoHandle = new GifInfoHandle(uri.getPath());
        } else {
            AssetFileDescriptor openAssetFileDescriptor = contentResolver.openAssetFileDescriptor(uri, "r");
            if (openAssetFileDescriptor != null) {
                gifInfoHandle = new GifInfoHandle(openAssetFileDescriptor);
            } else {
                throw new IOException(c.b.a.a.a.n("Could not open AssetFileDescriptor for ", uri));
            }
        }
    }

    public void a(long j) {
        if (this.m) {
            this.f6228d = 0L;
            this.n.sendEmptyMessageAtTime(-1, 0L);
            return;
        }
        ScheduledFuture<?> scheduledFuture = this.q;
        if (scheduledFuture != null) {
            scheduledFuture.cancel(false);
        }
        this.n.removeMessages(-1);
        this.q = this.f6226b.schedule(this.o, Math.max(j, 0L), TimeUnit.MILLISECONDS);
    }

    public final PorterDuffColorFilter b(ColorStateList colorStateList, PorterDuff.Mode mode) {
        if (colorStateList == null || mode == null) {
            return null;
        }
        return new PorterDuffColorFilter(colorStateList.getColorForState(getState(), 0), mode);
    }

    @Override // android.widget.MediaController.MediaPlayerControl
    public boolean canPause() {
        return true;
    }

    @Override // android.widget.MediaController.MediaPlayerControl
    public boolean canSeekBackward() {
        return this.f6232h.b() > 1;
    }

    @Override // android.widget.MediaController.MediaPlayerControl
    public boolean canSeekForward() {
        return this.f6232h.b() > 1;
    }

    @Override // android.graphics.drawable.Drawable
    public void draw(Canvas canvas) {
        boolean z;
        if (this.k == null || this.f6230f.getColorFilter() != null) {
            z = false;
        } else {
            this.f6230f.setColorFilter(this.k);
            z = true;
        }
        canvas.drawBitmap(this.f6231g, this.p, this.f6229e, this.f6230f);
        if (z) {
            this.f6230f.setColorFilter(null);
        }
    }

    @Override // android.graphics.drawable.Drawable
    public int getAlpha() {
        return this.f6230f.getAlpha();
    }

    @Override // android.widget.MediaController.MediaPlayerControl
    public int getAudioSessionId() {
        return 0;
    }

    @Override // android.widget.MediaController.MediaPlayerControl
    public int getBufferPercentage() {
        return 100;
    }

    @Override // android.graphics.drawable.Drawable
    public ColorFilter getColorFilter() {
        return this.f6230f.getColorFilter();
    }

    @Override // android.widget.MediaController.MediaPlayerControl
    public int getCurrentPosition() {
        int currentPosition;
        GifInfoHandle gifInfoHandle = this.f6232h;
        synchronized (gifInfoHandle) {
            currentPosition = GifInfoHandle.getCurrentPosition(gifInfoHandle.f6268b);
        }
        return currentPosition;
    }

    @Override // android.widget.MediaController.MediaPlayerControl
    public int getDuration() {
        int duration;
        GifInfoHandle gifInfoHandle = this.f6232h;
        synchronized (gifInfoHandle) {
            duration = GifInfoHandle.getDuration(gifInfoHandle.f6268b);
        }
        return duration;
    }

    @Override // android.graphics.drawable.Drawable
    public int getIntrinsicHeight() {
        return this.s;
    }

    @Override // android.graphics.drawable.Drawable
    public int getIntrinsicWidth() {
        return this.r;
    }

    @Override // android.graphics.drawable.Drawable
    public int getOpacity() {
        boolean isOpaque;
        GifInfoHandle gifInfoHandle = this.f6232h;
        synchronized (gifInfoHandle) {
            isOpaque = GifInfoHandle.isOpaque(gifInfoHandle.f6268b);
        }
        return (!isOpaque || this.f6230f.getAlpha() < 255) ? -2 : -1;
    }

    @Override // android.graphics.drawable.Drawable
    public void invalidateSelf() {
        super.invalidateSelf();
        if (this.m && this.f6227c) {
            long j = this.f6228d;
            if (j != Long.MIN_VALUE) {
                long max = Math.max(0L, j - SystemClock.uptimeMillis());
                this.f6228d = Long.MIN_VALUE;
                this.f6226b.remove(this.o);
                this.q = this.f6226b.schedule(this.o, max, TimeUnit.MILLISECONDS);
            }
        }
    }

    @Override // android.widget.MediaController.MediaPlayerControl
    public boolean isPlaying() {
        return this.f6227c;
    }

    @Override // android.graphics.drawable.Animatable
    public boolean isRunning() {
        return this.f6227c;
    }

    @Override // android.graphics.drawable.Drawable
    public boolean isStateful() {
        ColorStateList colorStateList;
        return super.isStateful() || ((colorStateList = this.j) != null && colorStateList.isStateful());
    }

    @Override // android.graphics.drawable.Drawable
    public void onBoundsChange(Rect rect) {
        this.f6229e.set(rect);
    }

    @Override // android.graphics.drawable.Drawable
    public boolean onStateChange(int[] iArr) {
        PorterDuff.Mode mode;
        ColorStateList colorStateList = this.j;
        if (colorStateList == null || (mode = this.l) == null) {
            return false;
        }
        this.k = b(colorStateList, mode);
        return true;
    }

    @Override // android.widget.MediaController.MediaPlayerControl
    public void pause() {
        stop();
    }

    @Override // android.widget.MediaController.MediaPlayerControl
    public void seekTo(int i) {
        if (i >= 0) {
            this.f6226b.execute(new a(this, i));
            return;
        }
        throw new IllegalArgumentException("Position is not positive");
    }

    @Override // android.graphics.drawable.Drawable
    public void setAlpha(int i) {
        this.f6230f.setAlpha(i);
    }

    @Override // android.graphics.drawable.Drawable
    public void setColorFilter(ColorFilter colorFilter) {
        this.f6230f.setColorFilter(colorFilter);
    }

    @Override // android.graphics.drawable.Drawable
    @Deprecated
    public void setDither(boolean z) {
        this.f6230f.setDither(z);
        invalidateSelf();
    }

    @Override // android.graphics.drawable.Drawable
    public void setFilterBitmap(boolean z) {
        this.f6230f.setFilterBitmap(z);
        invalidateSelf();
    }

    @Override // android.graphics.drawable.Drawable
    public void setTintList(ColorStateList colorStateList) {
        this.j = colorStateList;
        this.k = b(colorStateList, this.l);
        invalidateSelf();
    }

    @Override // android.graphics.drawable.Drawable
    public void setTintMode(PorterDuff.Mode mode) {
        this.l = mode;
        this.k = b(this.j, mode);
        invalidateSelf();
    }

    @Override // android.graphics.drawable.Drawable
    public boolean setVisible(boolean z, boolean z2) {
        boolean visible = super.setVisible(z, z2);
        if (!this.m) {
            if (z) {
                if (z2) {
                    this.f6226b.execute(new b(this, this));
                }
                if (visible) {
                    start();
                }
            } else if (visible) {
                stop();
            }
        }
        return visible;
    }

    @Override // android.graphics.drawable.Animatable, android.widget.MediaController.MediaPlayerControl
    public void start() {
        long restoreRemainder;
        synchronized (this) {
            if (this.f6227c) {
                return;
            }
            this.f6227c = true;
            GifInfoHandle gifInfoHandle = this.f6232h;
            synchronized (gifInfoHandle) {
                restoreRemainder = GifInfoHandle.restoreRemainder(gifInfoHandle.f6268b);
            }
            a(restoreRemainder);
        }
    }

    @Override // android.graphics.drawable.Animatable
    public void stop() {
        synchronized (this) {
            if (this.f6227c) {
                this.f6227c = false;
                ScheduledFuture<?> scheduledFuture = this.q;
                if (scheduledFuture != null) {
                    scheduledFuture.cancel(false);
                }
                this.n.removeMessages(-1);
                GifInfoHandle gifInfoHandle = this.f6232h;
                synchronized (gifInfoHandle) {
                    GifInfoHandle.saveRemainder(gifInfoHandle.f6268b);
                }
            }
        }
    }

    public String toString() {
        int nativeErrorCode;
        Locale locale = Locale.ENGLISH;
        Object[] objArr = new Object[4];
        objArr[0] = Integer.valueOf(this.f6232h.c());
        objArr[1] = Integer.valueOf(this.f6232h.a());
        objArr[2] = Integer.valueOf(this.f6232h.b());
        GifInfoHandle gifInfoHandle = this.f6232h;
        synchronized (gifInfoHandle) {
            nativeErrorCode = GifInfoHandle.getNativeErrorCode(gifInfoHandle.f6268b);
        }
        objArr[3] = Integer.valueOf(nativeErrorCode);
        return String.format(locale, "GIF: size: %dx%d, frames: %d, error: %d", objArr);
    }

    public c(Resources resources, int i) {
        this(new GifInfoHandle(resources.openRawResourceFd(i)), null, null, true);
        List<String> list = g.f6245a;
        TypedValue typedValue = new TypedValue();
        resources.getValue(i, typedValue, true);
        int i2 = typedValue.density;
        if (i2 == 0) {
            i2 = 160;
        } else if (i2 == 65535) {
            i2 = 0;
        }
        int i3 = resources.getDisplayMetrics().densityDpi;
        float f2 = (i2 <= 0 || i3 <= 0) ? 1.0f : i3 / i2;
        this.s = (int) (this.f6232h.a() * f2);
        this.r = (int) (this.f6232h.c() * f2);
    }

    public c(GifInfoHandle gifInfoHandle, c cVar, ScheduledThreadPoolExecutor scheduledThreadPoolExecutor, boolean z) {
        boolean isOpaque;
        this.f6227c = true;
        this.f6228d = Long.MIN_VALUE;
        this.f6229e = new Rect();
        this.f6230f = new Paint(6);
        this.i = new ConcurrentLinkedQueue<>();
        j jVar = new j(this);
        this.o = jVar;
        this.m = z;
        int i = e.f6242b;
        this.f6226b = e.b.f6243a;
        this.f6232h = gifInfoHandle;
        Bitmap createBitmap = Bitmap.createBitmap(gifInfoHandle.c(), gifInfoHandle.a(), Bitmap.Config.ARGB_8888);
        this.f6231g = createBitmap;
        synchronized (gifInfoHandle) {
            isOpaque = GifInfoHandle.isOpaque(gifInfoHandle.f6268b);
        }
        createBitmap.setHasAlpha(true ^ isOpaque);
        this.p = new Rect(0, 0, gifInfoHandle.c(), gifInfoHandle.a());
        this.n = new h(this);
        jVar.a();
        this.r = gifInfoHandle.c();
        this.s = gifInfoHandle.a();
    }
}