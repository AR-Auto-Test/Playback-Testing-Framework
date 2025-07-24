package c.d.d.a;

import android.os.Handler;
import com.google.mediapipe.components.FrameProcessor;

/* compiled from: lambda */
/* loaded from: classes2.dex */
public final /* synthetic */ class k implements FrameProcessor.ErrorListener {

    /* renamed from: a  reason: collision with root package name */
    public final /* synthetic */ Handler f4492a;

    /* renamed from: b  reason: collision with root package name */
    public final /* synthetic */ FrameProcessor.ErrorListener f4493b;

    public /* synthetic */ k(Handler handler, FrameProcessor.ErrorListener errorListener) {
        this.f4492a = handler;
        this.f4493b = errorListener;
    }

    @Override // com.google.mediapipe.components.FrameProcessor.ErrorListener
    public final void onError(final RuntimeException runtimeException) {
        Handler handler = this.f4492a;
        final FrameProcessor.ErrorListener errorListener = this.f4493b;
        handler.post(new Runnable() { // from class: c.d.d.a.l
            @Override // java.lang.Runnable
            public final void run() {
                FrameProcessor.ErrorListener.this.onError(runtimeException);
            }
        });
    }
}