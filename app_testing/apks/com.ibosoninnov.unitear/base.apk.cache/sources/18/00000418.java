package b.h.c;

import android.graphics.Canvas;
import android.graphics.Paint;
import android.graphics.Rect;
import android.graphics.Typeface;
import android.view.View;
import androidx.constraintlayout.widget.ConstraintLayout;

/* compiled from: Placeholder.java */
/* loaded from: classes.dex */
public class g extends View {

    /* renamed from: b  reason: collision with root package name */
    public int f2006b;

    /* renamed from: c  reason: collision with root package name */
    public View f2007c;

    /* renamed from: d  reason: collision with root package name */
    public int f2008d;

    public View getContent() {
        return this.f2007c;
    }

    public int getEmptyVisibility() {
        return this.f2008d;
    }

    @Override // android.view.View
    public void onDraw(Canvas canvas) {
        if (isInEditMode()) {
            canvas.drawRGB(223, 223, 223);
            Paint paint = new Paint();
            paint.setARGB(255, 210, 210, 210);
            paint.setTextAlign(Paint.Align.CENTER);
            paint.setTypeface(Typeface.create(Typeface.DEFAULT, 0));
            Rect rect = new Rect();
            canvas.getClipBounds(rect);
            paint.setTextSize(rect.height());
            int height = rect.height();
            int width = rect.width();
            paint.setTextAlign(Paint.Align.LEFT);
            paint.getTextBounds("?", 0, 1, rect);
            canvas.drawText("?", ((width / 2.0f) - (rect.width() / 2.0f)) - rect.left, ((rect.height() / 2.0f) + (height / 2.0f)) - rect.bottom, paint);
        }
    }

    public void setContentId(int i) {
        View findViewById;
        if (this.f2006b == i) {
            return;
        }
        View view = this.f2007c;
        if (view != null) {
            view.setVisibility(0);
            ((ConstraintLayout.a) this.f2007c.getLayoutParams()).a0 = false;
            this.f2007c = null;
        }
        this.f2006b = i;
        if (i == -1 || (findViewById = ((View) getParent()).findViewById(i)) == null) {
            return;
        }
        findViewById.setVisibility(8);
    }

    public void setEmptyVisibility(int i) {
        this.f2008d = i;
    }
}