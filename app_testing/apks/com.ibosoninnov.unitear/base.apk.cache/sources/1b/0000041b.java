package b.h.c;

import android.content.res.TypedArray;
import android.util.AttributeSet;
import android.view.View;
import android.view.ViewParent;
import androidx.constraintlayout.widget.ConstraintLayout;
import com.google.android.material.internal.StaticLayoutBuilderCompat;

/* compiled from: VirtualLayout.java */
/* loaded from: classes.dex */
public abstract class j extends b {
    public boolean i;
    public boolean j;

    @Override // b.h.c.b
    public void f(AttributeSet attributeSet) {
        super.f(attributeSet);
        if (attributeSet != null) {
            TypedArray obtainStyledAttributes = getContext().obtainStyledAttributes(attributeSet, i.f2010b);
            int indexCount = obtainStyledAttributes.getIndexCount();
            for (int i = 0; i < indexCount; i++) {
                int index = obtainStyledAttributes.getIndex(i);
                if (index == 6) {
                    this.i = true;
                } else if (index == 13) {
                    this.j = true;
                }
            }
            obtainStyledAttributes.recycle();
        }
    }

    @Override // b.h.c.b, android.view.View
    public void onAttachedToWindow() {
        ViewParent parent;
        super.onAttachedToWindow();
        if ((this.i || this.j) && (parent = getParent()) != null && (parent instanceof ConstraintLayout)) {
            ConstraintLayout constraintLayout = (ConstraintLayout) parent;
            int visibility = getVisibility();
            float elevation = getElevation();
            for (int i = 0; i < this.f1944c; i++) {
                View viewById = constraintLayout.getViewById(this.f1943b[i]);
                if (viewById != null) {
                    if (this.i) {
                        viewById.setVisibility(visibility);
                    }
                    if (this.j && elevation > StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD) {
                        viewById.setTranslationZ(viewById.getTranslationZ() + elevation);
                    }
                }
            }
        }
    }

    @Override // android.view.View
    public void setElevation(float f2) {
        super.setElevation(f2);
        d();
    }

    @Override // android.view.View
    public void setVisibility(int i) {
        super.setVisibility(i);
        d();
    }
}