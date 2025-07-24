package b.b.h;

import android.content.Context;
import android.graphics.Bitmap;
import android.util.AttributeSet;
import android.view.View;
import android.widget.RatingBar;
import com.ibosoninnov.unitear.R;

/* compiled from: AppCompatRatingBar.java */
/* loaded from: classes.dex */
public class s extends RatingBar {

    /* renamed from: b  reason: collision with root package name */
    public final q f919b;

    public s(Context context, AttributeSet attributeSet) {
        super(context, attributeSet, R.attr.ratingBarStyle);
        t0.a(this, getContext());
        q qVar = new q(this);
        this.f919b = qVar;
        qVar.a(attributeSet, R.attr.ratingBarStyle);
    }

    @Override // android.widget.RatingBar, android.widget.AbsSeekBar, android.widget.ProgressBar, android.view.View
    public synchronized void onMeasure(int i, int i2) {
        super.onMeasure(i, i2);
        Bitmap bitmap = this.f919b.f907c;
        if (bitmap != null) {
            setMeasuredDimension(View.resolveSizeAndState(bitmap.getWidth() * getNumStars(), i, 0), getMeasuredHeight());
        }
    }
}