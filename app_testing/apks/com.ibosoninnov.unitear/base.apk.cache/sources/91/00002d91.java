package h.a.a;

import java.util.concurrent.ScheduledThreadPoolExecutor;
import java.util.concurrent.ThreadPoolExecutor;

/* compiled from: GifRenderingExecutor.java */
/* loaded from: classes2.dex */
public final class e extends ScheduledThreadPoolExecutor {

    /* renamed from: b  reason: collision with root package name */
    public static final /* synthetic */ int f6242b = 0;

    /* compiled from: GifRenderingExecutor.java */
    /* loaded from: classes2.dex */
    public static final class b {

        /* renamed from: a  reason: collision with root package name */
        public static final e f6243a = new e(null);
    }

    public e(a aVar) {
        super(1, new ThreadPoolExecutor.DiscardPolicy());
    }
}