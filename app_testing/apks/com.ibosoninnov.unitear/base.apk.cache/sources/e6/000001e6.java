package b.b.h;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.drawable.Drawable;
import android.util.AttributeSet;
import android.widget.SeekBar;
import com.ibosoninnov.unitear.R;

/* compiled from: AppCompatSeekBar.java */
/* loaded from: classes.dex */
public class t extends SeekBar {

    /* renamed from: b  reason: collision with root package name */
    public final u f920b;

    public t(Context context, AttributeSet attributeSet) {
        super(context, attributeSet, R.attr.seekBarStyle);
        t0.a(this, getContext());
        u uVar = new u(this);
        this.f920b = uVar;
        uVar.a(attributeSet, R.attr.seekBarStyle);
    }

    @Override // android.widget.AbsSeekBar, android.widget.ProgressBar, android.view.View
    public void drawableStateChanged() {
        super.drawableStateChanged();
        u uVar = this.f920b;
        Drawable drawable = uVar.f929e;
        if (drawable != null && drawable.isStateful() && drawable.setState(uVar.f928d.getDrawableState())) {
            uVar.f928d.invalidateDrawable(drawable);
        }
    }

    @Override // android.widget.AbsSeekBar, android.widget.ProgressBar, android.view.View
    public void jumpDrawablesToCurrentState() {
        super.jumpDrawablesToCurrentState();
        Drawable drawable = this.f920b.f929e;
        if (drawable != null) {
            drawable.jumpToCurrentState();
        }
    }

    @Override // android.widget.AbsSeekBar, android.widget.ProgressBar, android.view.View
    public synchronized void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        this.f920b.d(canvas);
    }
}