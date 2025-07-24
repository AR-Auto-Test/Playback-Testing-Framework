package com.ibosoninnov.unitear;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Paint;
import android.graphics.PorterDuff;
import android.graphics.PorterDuffXfermode;
import android.graphics.RectF;
import android.util.AttributeSet;
import android.widget.RelativeLayout;
import b.j.c.a;
import com.google.android.material.internal.StaticLayoutBuilderCompat;

/* loaded from: classes2.dex */
public class CircleOverlayView extends RelativeLayout {

    /* renamed from: b  reason: collision with root package name */
    public Bitmap f5671b;

    public CircleOverlayView(Context context, AttributeSet attributeSet) {
        super(context, attributeSet);
    }

    @Override // android.view.ViewGroup, android.view.View
    public void dispatchDraw(Canvas canvas) {
        super.dispatchDraw(canvas);
        if (this.f5671b == null) {
            this.f5671b = Bitmap.createBitmap(getWidth(), getHeight(), Bitmap.Config.ARGB_8888);
            Canvas canvas2 = new Canvas(this.f5671b);
            RectF rectF = new RectF(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, getWidth(), getHeight());
            Paint paint = new Paint(1);
            Context context = getContext();
            Object obj = a.f2074a;
            paint.setColor(context.getColor(R.color.black));
            paint.setAlpha(150);
            canvas2.drawRect(rectF, paint);
            paint.setColor(0);
            paint.setXfermode(new PorterDuffXfermode(PorterDuff.Mode.SRC_OUT));
            canvas2.drawRoundRect(new RectF(((getRight() - getLeft()) / 15.0f) + getLeft(), ((getBottom() - getTop()) / 12.0f) + getTop(), getRight() - ((getRight() - getLeft()) / 15.0f), getBottom() - ((getBottom() - getTop()) / 5.0f)), 60.0f, 60.0f, paint);
        }
        canvas.drawBitmap(this.f5671b, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, (Paint) null);
    }

    @Override // android.view.View
    public boolean isInEditMode() {
        return true;
    }

    @Override // android.widget.RelativeLayout, android.view.ViewGroup, android.view.View
    public void onLayout(boolean z, int i, int i2, int i3, int i4) {
        super.onLayout(z, i, i2, i3, i4);
        this.f5671b = null;
    }
}