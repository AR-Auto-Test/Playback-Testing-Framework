package com.ibosoninnov.unitear.activities;

import android.annotation.SuppressLint;
import android.content.Context;
import android.content.Intent;
import android.os.Bundle;
import android.view.MotionEvent;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;
import androidx.cardview.widget.CardView;
import b.b.c.h;
import b.j.c.a;
import com.google.android.material.internal.StaticLayoutBuilderCompat;
import com.ibosoninnov.unitear.LoginWebviewActivity;
import com.ibosoninnov.unitear.R;
import com.ibosoninnov.unitear.activities.HelpActivity;
import java.util.Objects;

/* loaded from: classes2.dex */
public class HelpActivity extends h {
    public float A;
    public float B;
    public ImageView r;
    public ImageView s;
    public ScrollView t;
    public Button u;
    public Button v;
    public int w = 0;
    public LinearLayout x;
    public TextView y;
    public TextView z;

    @Override // android.app.Activity, android.view.Window.Callback
    public boolean dispatchTouchEvent(MotionEvent motionEvent) {
        onTouchEvent(motionEvent);
        return super.dispatchTouchEvent(motionEvent);
    }

    @Override // b.b.c.h, b.q.b.d, androidx.activity.ComponentActivity, b.j.b.e, android.app.Activity
    @SuppressLint({"ClickableViewAccessibility"})
    public void onCreate(Bundle bundle) {
        super.onCreate(bundle);
        setContentView(R.layout.activity_help);
        this.r = (ImageView) findViewById(R.id.backBtn);
        this.t = (ScrollView) findViewById(R.id.helpFrameLayout);
        this.u = (Button) findViewById(R.id.nextBtn);
        this.x = (LinearLayout) findViewById(R.id.linearLayout);
        this.v = (Button) findViewById(R.id.uploadBtn);
        this.s = (ImageView) findViewById(R.id.img);
        this.y = (TextView) findViewById(R.id.title);
        this.z = (TextView) findViewById(R.id.content);
        Context applicationContext = getApplicationContext();
        Object obj = a.f2074a;
        ((CardView) this.x.getChildAt(0)).setCardBackgroundColor(applicationContext.getColor(R.color.newBtn));
        this.y.setText(getResources().getString(R.string.help_fragment_title1));
        this.z.setText(getResources().getString(R.string.help_fragment_content1));
        this.s.setImageDrawable(getApplicationContext().getDrawable(R.drawable.ic_help_one_one));
        this.v.setOnClickListener(new View.OnClickListener() { // from class: c.e.b.df.f
            @Override // android.view.View.OnClickListener
            public final void onClick(View view) {
                HelpActivity helpActivity = HelpActivity.this;
                Objects.requireNonNull(helpActivity);
                helpActivity.startActivity(new Intent(helpActivity, LoginWebviewActivity.class));
            }
        });
        this.r.setOnClickListener(new View.OnClickListener() { // from class: c.e.b.df.g
            @Override // android.view.View.OnClickListener
            public final void onClick(View view) {
                HelpActivity.this.finish();
            }
        });
        this.t.setOnTouchListener(new View.OnTouchListener() { // from class: c.e.b.df.d
            @Override // android.view.View.OnTouchListener
            public final boolean onTouch(View view, MotionEvent motionEvent) {
                HelpActivity helpActivity = HelpActivity.this;
                Objects.requireNonNull(helpActivity);
                int action = motionEvent.getAction();
                if (action == 0) {
                    helpActivity.A = motionEvent.getX();
                } else if (action == 1) {
                    float x = motionEvent.getX();
                    helpActivity.B = x;
                    float f2 = helpActivity.A - x;
                    if (Math.abs(f2) > StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD) {
                        if (f2 >= StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD) {
                            int i = helpActivity.w;
                            if (i < 3) {
                                int i2 = i + 1;
                                helpActivity.w = i2;
                                helpActivity.v(i2);
                            } else {
                                helpActivity.finish();
                            }
                        } else {
                            int i3 = helpActivity.w;
                            if (i3 > 0) {
                                int i4 = i3 - 1;
                                helpActivity.w = i4;
                                helpActivity.v(i4);
                            }
                        }
                    }
                    for (int i5 = 0; i5 < helpActivity.x.getChildCount(); i5++) {
                        CardView cardView = (CardView) helpActivity.x.getChildAt(i5);
                        if (i5 == helpActivity.w) {
                            Context applicationContext2 = helpActivity.getApplicationContext();
                            Object obj2 = b.j.c.a.f2074a;
                            cardView.setCardBackgroundColor(applicationContext2.getColor(R.color.newBtn));
                        } else {
                            Context applicationContext3 = helpActivity.getApplicationContext();
                            Object obj3 = b.j.c.a.f2074a;
                            cardView.setCardBackgroundColor(applicationContext3.getColor(R.color.gray11));
                        }
                    }
                }
                return false;
            }
        });
        this.u.setOnClickListener(new View.OnClickListener() { // from class: c.e.b.df.e
            @Override // android.view.View.OnClickListener
            public final void onClick(View view) {
                HelpActivity helpActivity = HelpActivity.this;
                int i = helpActivity.w + 1;
                helpActivity.w = i;
                helpActivity.v(i);
                for (int i2 = 0; i2 < helpActivity.x.getChildCount(); i2++) {
                    CardView cardView = (CardView) helpActivity.x.getChildAt(i2);
                    if (i2 == helpActivity.w) {
                        Context applicationContext2 = helpActivity.getApplicationContext();
                        Object obj2 = b.j.c.a.f2074a;
                        cardView.setCardBackgroundColor(applicationContext2.getColor(R.color.newBtn));
                    } else {
                        Context applicationContext3 = helpActivity.getApplicationContext();
                        Object obj3 = b.j.c.a.f2074a;
                        cardView.setCardBackgroundColor(applicationContext3.getColor(R.color.gray11));
                    }
                }
            }
        });
    }

    public final void v(int i) {
        if (i == 4) {
            finish();
            return;
        }
        if (i == 0) {
            this.y.setText(getResources().getString(R.string.help_fragment_title1));
            this.z.setText(getResources().getString(R.string.help_fragment_content1));
            ImageView imageView = this.s;
            Context applicationContext = getApplicationContext();
            Object obj = a.f2074a;
            imageView.setImageDrawable(applicationContext.getDrawable(R.drawable.ic_help_one_one));
        } else if (i == 1) {
            this.y.setText(getResources().getString(R.string.help_fragment_title2));
            this.z.setText(getResources().getString(R.string.help_fragment_content2));
            ImageView imageView2 = this.s;
            Context applicationContext2 = getApplicationContext();
            Object obj2 = a.f2074a;
            imageView2.setImageDrawable(applicationContext2.getDrawable(R.drawable.ic_help_one_two));
        } else if (i == 2) {
            this.y.setText(getResources().getString(R.string.help_fragment_title3));
            this.z.setText(getResources().getString(R.string.help_fragment_content3));
            ImageView imageView3 = this.s;
            Context applicationContext3 = getApplicationContext();
            Object obj3 = a.f2074a;
            imageView3.setImageDrawable(applicationContext3.getDrawable(R.drawable.ic_help_one_three));
        } else if (i == 3) {
            this.y.setText(getResources().getString(R.string.help_fragment_title4));
            this.z.setText(getResources().getString(R.string.help_fragment_content4));
            ImageView imageView4 = this.s;
            Context applicationContext4 = getApplicationContext();
            Object obj4 = a.f2074a;
            imageView4.setImageDrawable(applicationContext4.getDrawable(R.drawable.help_one_four));
        }
        if (i > 0) {
            this.v.setVisibility(8);
        } else {
            this.v.setVisibility(0);
        }
        if (i == 3) {
            this.u.setText(getResources().getString(R.string.finish));
        } else {
            this.u.setText(getResources().getString(R.string.next));
        }
    }
}