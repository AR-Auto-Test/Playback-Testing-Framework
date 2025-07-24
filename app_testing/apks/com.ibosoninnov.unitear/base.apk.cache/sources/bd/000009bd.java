package c.e.b;

import android.animation.Animator;
import android.animation.AnimatorListenerAdapter;
import com.ibosoninnov.unitear.LoginWebviewActivity;

/* compiled from: LoginWebviewActivity.java */
/* loaded from: classes2.dex */
public class ce extends AnimatorListenerAdapter {

    /* renamed from: a  reason: collision with root package name */
    public final /* synthetic */ LoginWebviewActivity f4618a;

    public ce(LoginWebviewActivity loginWebviewActivity) {
        this.f4618a = loginWebviewActivity;
    }

    @Override // android.animation.AnimatorListenerAdapter, android.animation.Animator.AnimatorListener
    public void onAnimationEnd(Animator animator) {
        super.onAnimationEnd(animator);
        this.f4618a.u.setVisibility(8);
    }
}