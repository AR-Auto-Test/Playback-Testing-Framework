package c.d.c.i.a;

import com.google.firebase.encoders.EncodingException;
import com.google.firebase.encoders.ObjectEncoder;
import com.google.firebase.encoders.ObjectEncoderContext;
import com.google.firebase.encoders.json.JsonDataEncoderBuilder;

/* compiled from: lambda */
/* loaded from: classes2.dex */
public final /* synthetic */ class a implements ObjectEncoder {

    /* renamed from: a  reason: collision with root package name */
    public static final /* synthetic */ a f4440a = new a();

    /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object, java.lang.Object] */
    @Override // com.google.firebase.encoders.Encoder
    public final void encode(Object obj, ObjectEncoderContext objectEncoderContext) {
        int i = JsonDataEncoderBuilder.f5645a;
        StringBuilder x = c.b.a.a.a.x("Couldn't find encoder for type ");
        x.append(obj.getClass().getCanonicalName());
        throw new EncodingException(x.toString());
    }
}