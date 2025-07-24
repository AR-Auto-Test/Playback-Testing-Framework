package b.m;

import android.view.Choreographer;
import androidx.databinding.ViewDataBinding;

/* compiled from: ViewDataBinding.java */
/* loaded from: classes.dex */
public class j implements Choreographer.FrameCallback {

    /* renamed from: b  reason: collision with root package name */
    public final /* synthetic */ ViewDataBinding f2337b;

    public j(ViewDataBinding viewDataBinding) {
        this.f2337b = viewDataBinding;
    }

    @Override // android.view.Choreographer.FrameCallback
    public void doFrame(long j) {
        this.f2337b.f262h.run();
    }
}