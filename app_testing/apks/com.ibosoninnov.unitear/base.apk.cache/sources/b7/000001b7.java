package b.b.h;

import android.annotation.SuppressLint;
import android.content.Context;
import android.content.res.ColorStateList;
import android.graphics.PorterDuff;
import android.graphics.drawable.Drawable;
import android.util.AttributeSet;
import android.view.ActionMode;
import android.view.accessibility.AccessibilityEvent;
import android.view.accessibility.AccessibilityNodeInfo;
import android.widget.Button;
import com.ibosoninnov.unitear.R;

/* compiled from: AppCompatButton.java */
/* loaded from: classes.dex */
public class f extends Button implements b.j.k.b {
    private final e mBackgroundTintHelper;
    private final y mTextHelper;

    public f(Context context) {
        this(context, null);
    }

    @Override // android.widget.TextView, android.view.View
    public void drawableStateChanged() {
        super.drawableStateChanged();
        e eVar = this.mBackgroundTintHelper;
        if (eVar != null) {
            eVar.a();
        }
        y yVar = this.mTextHelper;
        if (yVar != null) {
            yVar.b();
        }
    }

    @Override // android.widget.TextView
    public int getAutoSizeMaxTextSize() {
        if (b.j.k.b.f2297a) {
            return super.getAutoSizeMaxTextSize();
        }
        y yVar = this.mTextHelper;
        if (yVar != null) {
            return Math.round(yVar.i.f786h);
        }
        return -1;
    }

    @Override // android.widget.TextView
    public int getAutoSizeMinTextSize() {
        if (b.j.k.b.f2297a) {
            return super.getAutoSizeMinTextSize();
        }
        y yVar = this.mTextHelper;
        if (yVar != null) {
            return Math.round(yVar.i.f785g);
        }
        return -1;
    }

    @Override // android.widget.TextView
    public int getAutoSizeStepGranularity() {
        if (b.j.k.b.f2297a) {
            return super.getAutoSizeStepGranularity();
        }
        y yVar = this.mTextHelper;
        if (yVar != null) {
            return Math.round(yVar.i.f784f);
        }
        return -1;
    }

    @Override // android.widget.TextView
    public int[] getAutoSizeTextAvailableSizes() {
        if (b.j.k.b.f2297a) {
            return super.getAutoSizeTextAvailableSizes();
        }
        y yVar = this.mTextHelper;
        return yVar != null ? yVar.i.i : new int[0];
    }

    @Override // android.widget.TextView
    @SuppressLint({"WrongConstant"})
    public int getAutoSizeTextType() {
        if (b.j.k.b.f2297a) {
            return super.getAutoSizeTextType() == 1 ? 1 : 0;
        }
        y yVar = this.mTextHelper;
        if (yVar != null) {
            return yVar.i.f782d;
        }
        return 0;
    }

    public ColorStateList getSupportBackgroundTintList() {
        e eVar = this.mBackgroundTintHelper;
        if (eVar != null) {
            return eVar.b();
        }
        return null;
    }

    public PorterDuff.Mode getSupportBackgroundTintMode() {
        e eVar = this.mBackgroundTintHelper;
        if (eVar != null) {
            return eVar.c();
        }
        return null;
    }

    public ColorStateList getSupportCompoundDrawablesTintList() {
        w0 w0Var = this.mTextHelper.f966h;
        if (w0Var != null) {
            return w0Var.f953a;
        }
        return null;
    }

    public PorterDuff.Mode getSupportCompoundDrawablesTintMode() {
        w0 w0Var = this.mTextHelper.f966h;
        if (w0Var != null) {
            return w0Var.f954b;
        }
        return null;
    }

    @Override // android.view.View
    public void onInitializeAccessibilityEvent(AccessibilityEvent accessibilityEvent) {
        super.onInitializeAccessibilityEvent(accessibilityEvent);
        accessibilityEvent.setClassName(Button.class.getName());
    }

    @Override // android.view.View
    public void onInitializeAccessibilityNodeInfo(AccessibilityNodeInfo accessibilityNodeInfo) {
        super.onInitializeAccessibilityNodeInfo(accessibilityNodeInfo);
        accessibilityNodeInfo.setClassName(Button.class.getName());
    }

    @Override // android.widget.TextView, android.view.View
    public void onLayout(boolean z, int i, int i2, int i3, int i4) {
        super.onLayout(z, i, i2, i3, i4);
        y yVar = this.mTextHelper;
        if (yVar == null || b.j.k.b.f2297a) {
            return;
        }
        yVar.i.a();
    }

    @Override // android.widget.TextView
    public void onTextChanged(CharSequence charSequence, int i, int i2, int i3) {
        super.onTextChanged(charSequence, i, i2, i3);
        y yVar = this.mTextHelper;
        if (yVar == null || b.j.k.b.f2297a || !yVar.d()) {
            return;
        }
        this.mTextHelper.i.a();
    }

    @Override // android.widget.TextView
    public void setAutoSizeTextTypeUniformWithConfiguration(int i, int i2, int i3, int i4) {
        if (b.j.k.b.f2297a) {
            super.setAutoSizeTextTypeUniformWithConfiguration(i, i2, i3, i4);
            return;
        }
        y yVar = this.mTextHelper;
        if (yVar != null) {
            yVar.g(i, i2, i3, i4);
        }
    }

    @Override // android.widget.TextView
    public void setAutoSizeTextTypeUniformWithPresetSizes(int[] iArr, int i) {
        if (b.j.k.b.f2297a) {
            super.setAutoSizeTextTypeUniformWithPresetSizes(iArr, i);
            return;
        }
        y yVar = this.mTextHelper;
        if (yVar != null) {
            yVar.h(iArr, i);
        }
    }

    @Override // android.widget.TextView
    public void setAutoSizeTextTypeWithDefaults(int i) {
        if (b.j.k.b.f2297a) {
            super.setAutoSizeTextTypeWithDefaults(i);
            return;
        }
        y yVar = this.mTextHelper;
        if (yVar != null) {
            yVar.i(i);
        }
    }

    @Override // android.view.View
    public void setBackgroundDrawable(Drawable drawable) {
        super.setBackgroundDrawable(drawable);
        e eVar = this.mBackgroundTintHelper;
        if (eVar != null) {
            eVar.e();
        }
    }

    @Override // android.view.View
    public void setBackgroundResource(int i) {
        super.setBackgroundResource(i);
        e eVar = this.mBackgroundTintHelper;
        if (eVar != null) {
            eVar.f(i);
        }
    }

    @Override // android.widget.TextView
    public void setCustomSelectionActionModeCallback(ActionMode.Callback callback) {
        super.setCustomSelectionActionModeCallback(b.j.b.d.U(this, callback));
    }

    public void setSupportAllCaps(boolean z) {
        y yVar = this.mTextHelper;
        if (yVar != null) {
            yVar.f959a.setAllCaps(z);
        }
    }

    public void setSupportBackgroundTintList(ColorStateList colorStateList) {
        e eVar = this.mBackgroundTintHelper;
        if (eVar != null) {
            eVar.h(colorStateList);
        }
    }

    public void setSupportBackgroundTintMode(PorterDuff.Mode mode) {
        e eVar = this.mBackgroundTintHelper;
        if (eVar != null) {
            eVar.i(mode);
        }
    }

    public void setSupportCompoundDrawablesTintList(ColorStateList colorStateList) {
        this.mTextHelper.j(colorStateList);
        this.mTextHelper.b();
    }

    public void setSupportCompoundDrawablesTintMode(PorterDuff.Mode mode) {
        this.mTextHelper.k(mode);
        this.mTextHelper.b();
    }

    @Override // android.widget.TextView
    public void setTextAppearance(Context context, int i) {
        super.setTextAppearance(context, i);
        y yVar = this.mTextHelper;
        if (yVar != null) {
            yVar.f(context, i);
        }
    }

    @Override // android.widget.TextView
    public void setTextSize(int i, float f2) {
        boolean z = b.j.k.b.f2297a;
        if (z) {
            super.setTextSize(i, f2);
            return;
        }
        y yVar = this.mTextHelper;
        if (yVar == null || z || yVar.d()) {
            return;
        }
        yVar.i.f(i, f2);
    }

    public f(Context context, AttributeSet attributeSet) {
        this(context, attributeSet, R.attr.buttonStyle);
    }

    /* JADX WARN: 'super' call moved to the top of the method (can break code semantics) */
    public f(Context context, AttributeSet attributeSet, int i) {
        super(context, attributeSet, i);
        v0.a(context);
        t0.a(this, getContext());
        e eVar = new e(this);
        this.mBackgroundTintHelper = eVar;
        eVar.d(attributeSet, i);
        y yVar = new y(this);
        this.mTextHelper = yVar;
        yVar.e(attributeSet, i);
        yVar.b();
    }
}