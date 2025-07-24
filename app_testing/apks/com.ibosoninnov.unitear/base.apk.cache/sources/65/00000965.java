package c.d.c.i.b;

import com.google.firebase.encoders.EncodingException;
import com.google.firebase.encoders.ObjectEncoder;
import com.google.firebase.encoders.ObjectEncoderContext;
import com.google.firebase.encoders.proto.ProtobufEncoder;

/* compiled from: lambda */
/* loaded from: classes2.dex */
public final /* synthetic */ class b implements ObjectEncoder {

    /* renamed from: a  reason: collision with root package name */
    public static final /* synthetic */ b f4444a = new b();

    /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object, java.lang.Object] */
    @Override // com.google.firebase.encoders.Encoder
    public final void encode(Object obj, ObjectEncoderContext objectEncoderContext) {
        int i = ProtobufEncoder.Builder.f5646a;
        StringBuilder x = c.b.a.a.a.x("Couldn't find encoder for type ");
        x.append(obj.getClass().getCanonicalName());
        throw new EncodingException(x.toString());
    }
}